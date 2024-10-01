# package imports
from typing import Dict
import torch
import torch.nn as nn

import rospy

# Local imports
from scripts.mapper.model import sample_pdf, batchify, get_sdf_loss, mse2psnr, compute_loss
from scripts.mapper.model.decoder import ColorSDFNet, ColorSDFNet_v2
from scripts.mapper.model.encodings import get_encoder

class JointEncoding(nn.Module):
    def __init__(self, config:dict, bound_box:torch.Tensor, depth_max:float):
        super(JointEncoding, self).__init__()
        self.__config = config
        self.__bounding_box = bound_box
        
        self.__dataset_near = self.__config['dataset']['near']
        self.__dataset_far = self.__config['dataset']['far']
        self.__sc_factor = self.__config['dataset']['sc_factor']
        self.__voxel_sdf = self.__config['mapper']['scene_rep']['grid']['voxel_sdf']
        self.__voxel_color = self.__config['mapper']['scene_rep']['grid']['voxel_color']
        self.__pos_enc = self.__config['mapper']['scene_rep']['pos']['enc']
        self.__pos_n_bins = self.__config['mapper']['scene_rep']['pos']['n_bins']
        self.__grid_enc = self.__config['mapper']['scene_rep']['grid']['enc']
        self.__grid_hash_size = self.__config['mapper']['scene_rep']['grid']['hash_size']
        self.__one_grid = self.__config['mapper']['scene_rep']['grid']['oneGrid']
        self.__tcnn_encoding = self.__config['mapper']['scene_rep']['grid']['tcnn_encoding']
        self.__trunc = self.__config['mapper']['scene_rep']['training']['trunc']
        self.__n_range_d = self.__config['mapper']['scene_rep']['training']['n_range_d']
        self.__white_bkgd = self.__config['mapper']['scene_rep']['training']['white_bkgd']
        self.__range_d = self.__config['mapper']['scene_rep']['training']['range_d']
        self.__n_samples_d = self.__config['mapper']['scene_rep']['training']['n_samples_d']
        self.__n_samples = self.__config['mapper']['scene_rep']['training']['n_samples']
        self.__perturb = self.__config['mapper']['scene_rep']['training']['perturb']
        self.__n_importance = self.__config['mapper']['scene_rep']['training']['n_importance']
        self.__depth_trunc = self.__config['mapper']['scene_rep']['training']['depth_trunc']
        if self.__depth_trunc is None:
            self.__depth_trunc = depth_max
        self.__rgb_missing = self.__config['mapper']['scene_rep']['training']['rgb_missing']
        
        self.__get_resolution()
        self.__get_encoding()
        self.__get_decoder()

    def __get_resolution(self):
        '''
        Get the resolution of the grid
        '''
        dim_max = (self.__bounding_box[:,1] - self.__bounding_box[:,0]).max()
        if self.__voxel_sdf > 10:
            self.resolution_sdf = self.__voxel_sdf
        else:
            self.resolution_sdf = int(dim_max / self.__voxel_sdf)
        
        if self.__voxel_color > 10:
            self.resolution_color = self.__voxel_color
        else:
            self.resolution_color = int(dim_max / self.__voxel_color)
        
        rospy.logdebug(f'SDF resolution: {self.resolution_sdf}')

    def __get_encoding(self):
        '''
        Get the encoding of the scene representation
        '''
        # Coordinate encoding
        self.embedpos_fn, self.input_ch_pos = get_encoder(self.__pos_enc, n_bins=self.__pos_n_bins)

        # Sparse parametric encoding (SDF)
        self.embed_fn, self.input_ch = get_encoder(self.__grid_enc, log2_hashmap_size=self.__grid_hash_size, desired_resolution=self.resolution_sdf)

        # Sparse parametric encoding (Color)
        if not self.__one_grid:
            rospy.logdebug(f'Color resolution: {self.resolution_color}')
            self.embed_fn_color, self.input_ch_color = get_encoder(self.__grid_enc, log2_hashmap_size=self.__grid_hash_size, desired_resolution=self.resolution_color)

    def __get_decoder(self):
        '''
        Get the decoder of the scene representation
        '''
        if not self.__one_grid:
            self.decoder = ColorSDFNet(self.__config, input_ch=self.input_ch, input_ch_pos=self.input_ch_pos)
        else:
            self.decoder = ColorSDFNet_v2(self.__config, input_ch=self.input_ch, input_ch_pos=self.input_ch_pos)
        
        self.color_net = batchify(self.decoder.color_net, None)
        self.sdf_net = batchify(self.decoder.sdf_net, None)

    def __sdf2weights(self, sdf, z_vals):
        '''
        Convert signed distance function to weights.

        Params:
            sdf: [N_rays, N_samples]
            z_vals: [N_rays, N_samples]
        Returns:
            weights: [N_rays, N_samples]
        '''
        weights = torch.sigmoid(sdf / self.__trunc) * torch.sigmoid(-sdf / self.__trunc)

        signs = sdf[:, 1:] * sdf[:, :-1]
        mask = torch.where(signs < 0.0, torch.ones_like(signs), torch.zeros_like(signs))
        inds = torch.argmax(mask, axis=1)
        inds = inds[..., None]
        z_min = torch.gather(z_vals, 1, inds) # The first surface
        mask = torch.where(z_vals < z_min + self.__sc_factor * self.__trunc, torch.ones_like(z_vals), torch.zeros_like(z_vals))

        weights = weights * mask
        return weights / (torch.sum(weights, axis=-1, keepdims=True) + 1e-8)
    
    def __raw2outputs(self, raw, z_vals, white_bkgd=False):
        '''
        Perform volume rendering using weights computed from sdf.

        Params:
            raw: [N_rays, N_samples, 4]
            z_vals: [N_rays, N_samples]
        Returns:
            rgb_map: [N_rays, 3]
            disp_map: [N_rays]
            acc_map: [N_rays]
            weights: [N_rays, N_samples]
        '''
        rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
        weights = self.__sdf2weights(raw[..., 3], z_vals)
        rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]

        depth_map = torch.sum(weights * z_vals, -1)
        depth_var = torch.sum(weights * torch.square(z_vals - depth_map.unsqueeze(-1)), dim=-1)
        disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
        acc_map = torch.sum(weights, -1)

        if white_bkgd:
            rgb_map = rgb_map + (1.-acc_map[...,None])

        return rgb_map, disp_map, acc_map, weights, depth_map, depth_var
    
    def query_sdf(self, query_points, return_geo=False, embed=False):
        '''
        Get the SDF value of the query points
        Params:
            query_points: [N_rays, N_samples, 3]
        Returns:
            sdf: [N_rays, N_samples]
            geo_feat: [N_rays, N_samples, channel]
        '''
        inputs_flat = torch.reshape(query_points, [-1, query_points.shape[-1]])
  
        embedded = self.embed_fn(inputs_flat)
        if embed:
            return torch.reshape(embedded, list(query_points.shape[:-1]) + [embedded.shape[-1]])

        embedded_pos = self.embedpos_fn(inputs_flat)
        out = self.sdf_net(torch.cat([embedded, embedded_pos], dim=-1))
        sdf, geo_feat = out[..., :1], out[..., 1:]

        sdf = torch.reshape(sdf, list(query_points.shape[:-1]))
        if not return_geo:
            return sdf
        geo_feat = torch.reshape(geo_feat, list(query_points.shape[:-1]) + [geo_feat.shape[-1]])

        return sdf, geo_feat
    
    def query_color(self, query_points):
        return torch.sigmoid(self.query_color_sdf(query_points)[..., :3])
      
    def query_color_sdf(self, query_points):
        '''
        Query the color and sdf at query_points.

        Params:
            query_points: [N_rays, N_samples, 3]
        Returns:
            raw: [N_rays, N_samples, 4]
        '''
        inputs_flat = torch.reshape(query_points, [-1, query_points.shape[-1]])

        embed = self.embed_fn(inputs_flat)
        embe_pos = self.embedpos_fn(inputs_flat)
        if not self.__one_grid:
            embed_color = self.embed_fn_color(inputs_flat)
            return self.decoder(embed, embe_pos, embed_color)
        return self.decoder(embed, embe_pos)
    
    def tcnn_encoding(self, inputs:torch.Tensor) -> torch.Tensor:
        if self.__tcnn_encoding:
            # assert torch.all(inputs >= self.__bounding_box[:, 0]) and torch.all(inputs <= self.__bounding_box[:, 1]), 'Input out of bound'
            return (inputs - self.__bounding_box[:, 0]) / (self.__bounding_box[:, 1] - self.__bounding_box[:, 0])
        return inputs
    
    def run_network(self, inputs:torch.Tensor) -> torch.Tensor:
        """
        Run the network on a batch of inputs.

        Params:
            inputs: [N_rays, N_samples, 3]
        Returns:
            outputs: [N_rays, N_samples, 4]
        """
        inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
        
        # Normalize the input to [0, 1] (TCNN convention)
        inputs_flat = self.tcnn_encoding(inputs_flat)

        outputs_flat = batchify(self.query_color_sdf, None)(inputs_flat)
        outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])

        return outputs
    
    def render_surface_color(self, rays_o:torch.Tensor, normal:torch.Tensor):
        '''
        Render the surface color of the points.
        Params:
            points: [N_rays, 1, 3]
            normal: [N_rays, 3]
        '''
        n_rays = rays_o.shape[0]
        z_vals = torch.linspace(-self.__trunc, self.__trunc, steps=self.__n_range_d).to(rays_o)
        z_vals = z_vals.repeat(n_rays, 1)
        # Run rendering pipeline
        
        pts = rays_o[...,:] + normal[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]
        raw = self.run_network(pts)
        rgb, disp_map, acc_map, weights, depth_map, depth_var = self.__raw2outputs(raw, z_vals, self.__white_bkgd)
        return rgb
    
    def render_rays(self, rays_o:torch.Tensor, rays_d:torch.Tensor, target_d:torch.Tensor=None) -> Dict[str, torch.Tensor]:
        '''
        Params:
            rays_o: [N_rays, 3]
            rays_d: [N_rays, 3]
            target_d: [N_rays, 1]

        '''
        n_rays = rays_o.shape[0]

        # Sample depth
        if target_d is not None:
            z_samples = torch.linspace(-self.__range_d, self.__range_d, steps=self.__n_range_d).to(target_d) 
            z_samples = z_samples[None, :].repeat(n_rays, 1) + target_d
            z_samples[target_d.squeeze()<=0] = torch.linspace(self.__dataset_near, self.__dataset_far, steps=self.__n_range_d).to(target_d) 

            if self.__n_samples_d > 0:
                z_vals = torch.linspace(self.__dataset_near, self.__dataset_far, self.__n_samples_d)[None, :].repeat(n_rays, 1).to(rays_o)
                z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
            else:
                z_vals = z_samples
        else:
            z_vals = torch.linspace(self.__dataset_near, self.__dataset_far, self.__n_samples).to(rays_o)
            z_vals = z_vals[None, :].repeat(n_rays, 1) # [n_rays, n_samples]

        # Perturb sampling depths
        if self.__perturb > 0.:
            mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
            upper = torch.cat([mids, z_vals[...,-1:]], -1)
            lower = torch.cat([z_vals[...,:1], mids], -1)
            z_vals = lower + (upper - lower) * torch.rand(z_vals.shape).to(rays_o)

        # Run rendering pipeline
        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]
        raw = self.run_network(pts)
        rgb_map, disp_map, acc_map, weights, depth_map, depth_var = self.__raw2outputs(raw, z_vals, self.__white_bkgd)

        # Importance sampling
        if self.__n_importance > 0:

            rgb_map_0, disp_map_0, acc_map_0, depth_map_0, depth_var_0 = rgb_map, disp_map, acc_map, depth_map, depth_var

            z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
            z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], self.__n_importance, det=(self.__perturb==0.))
            z_samples = z_samples.detach()

            z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
            pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples + N_importance, 3]

            raw = self.run_network(pts)
            rgb_map, disp_map, acc_map, weights, depth_map, depth_var = self.__raw2outputs(raw, z_vals, self.__white_bkgd)

        # Return rendering outputs
        ret = {'rgb' : rgb_map, 'depth' :depth_map, 
               'disp_map' : disp_map, 'acc_map' : acc_map, 
               'depth_var':depth_var,}
        ret = {**ret, 'z_vals': z_vals}

        ret['raw'] = raw

        if self.__n_importance > 0:
            ret['rgb0'] = rgb_map_0
            ret['disp0'] = disp_map_0
            ret['acc0'] = acc_map_0
            ret['depth0'] = depth_map_0
            ret['depth_var0'] = depth_var_0
            ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)

        return ret
    
    def forward(self, rays_o:torch.Tensor, rays_d:torch.Tensor, target_rgb:torch.Tensor, target_d:torch.Tensor, global_step:int=0):
        '''
        Params:
            rays_o: ray origins (Bs, 3)
            rays_d: ray directions (Bs, 3)
            frame_ids: use for pose correction (Bs, 1)
            target_rgb: rgb value (Bs, 3)
            target_d: depth value (Bs, 1)
            c2w_array: poses (N, 4, 4) 
             r r r tx
             r r r ty
             r r r tz
        '''

        # Get render results
        rend_dict = self.render_rays(rays_o, rays_d, target_d=target_d)

        if not self.training:
            return rend_dict
        
        # Get depth and rgb weights for loss
        valid_depth_mask:torch.Tensor = (target_d.squeeze() > 0.) * (target_d.squeeze() < self.__depth_trunc)
        rgb_weight = valid_depth_mask.clone().unsqueeze(-1)
        rgb_weight[rgb_weight==0] = self.__rgb_missing

        # Get render loss
        rgb_loss = compute_loss(rend_dict["rgb"]*rgb_weight, target_rgb*rgb_weight)
        psnr = mse2psnr(rgb_loss)
        depth_loss = compute_loss(rend_dict["depth"].squeeze()[valid_depth_mask], target_d.squeeze()[valid_depth_mask])

        if 'rgb0' in rend_dict:
            rgb_loss += compute_loss(rend_dict["rgb0"]*rgb_weight, target_rgb*rgb_weight)
            depth_loss += compute_loss(rend_dict["depth0"][valid_depth_mask], target_d.squeeze()[valid_depth_mask])
        
        # Get sdf loss
        z_vals = rend_dict['z_vals']  # [N_rand, N_samples + N_importance]
        sdf = rend_dict['raw'][..., -1]  # [N_rand, N_samples + N_importance]
        truncation = self.__trunc * self.__sc_factor
        fs_loss, sdf_loss = get_sdf_loss(
            z_vals,
            target_d,
            sdf,
            truncation,
            'l2',
            grad=None)         
        

        ret = {
            "rgb": rend_dict["rgb"],
            "depth": rend_dict["depth"],
            "rgb_loss": rgb_loss,
            "depth_loss": depth_loss,
            "sdf_loss": sdf_loss,
            "fs_loss": fs_loss,
            "psnr": psnr,
        }

        return ret

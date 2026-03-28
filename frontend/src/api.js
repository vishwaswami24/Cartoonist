import axios from 'axios';

const API_BASE_URL = '/api';

export const api = {
  // Cartoonize an image using Pix2Pix
  cartoonizeImage: async (imageFile) => {
    const formData = new FormData();
    formData.append('image', imageFile);
    
    const response = await axios.post(`${API_BASE_URL}/cartoonize`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      responseType: 'blob'
    });
    
    return URL.createObjectURL(response.data);
  },

  // Generate cartoon faces using StyleGAN
  generateFaces: async (numImages = 4) => {
    const response = await axios.post(`${API_BASE_URL}/generate`, {
      num_images: numImages
    });
    
    return response.data.images;
  },

  // Interpolate between two latent vectors
  interpolate: async (seed1, seed2, numSteps = 10) => {
    const response = await axios.post(`${API_BASE_URL}/interpolate`, {
      seed1,
      seed2,
      num_steps: numSteps
    });
    
    return response.data.images;
  },

  // Check if models are loaded
  checkStatus: async () => {
    const response = await axios.get(`${API_BASE_URL}/status`);
    return response.data;
  }
};

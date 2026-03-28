import { useState, useEffect } from 'react';
import { useDropzone } from 'react-dropzone';
import axios from 'axios';
import './App.css';

function App() {
  const [activeTab, setActiveTab] = useState('cartoonize');
  const [selectedImage, setSelectedImage] = useState(null);
  const [resultImage, setResultImage] = useState(null);
  const [loading, setLoading] = useState(false);
  const [generatedImages, setGeneratedImages] = useState([]);
  const [numImages, setNumImages] = useState(4);
  const [status, setStatus] = useState({ pix2pix: false, stylegan: false });
  const [imagePreview, setImagePreview] = useState(null);

  // Check backend status
  useEffect(() => {
    checkStatus();
  }, []);

  const checkStatus = async () => {
    try {
      const response = await axios.get('/api/status');
      setStatus(response.data);
    } catch (error) {
      console.error('Backend not available');
    }
  };

  const onDrop = (acceptedFiles) => {
    if (acceptedFiles.length > 0) {
      const file = acceptedFiles[0];
      setSelectedImage(file);
      setResultImage(null);
      
      // Create preview URL
      const previewUrl = URL.createObjectURL(file);
      setImagePreview(previewUrl);
    }
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.png', '.jpg', '.jpeg']
    },
    multiple: false
  });

  const handleCartoonize = async () => {
    if (!selectedImage) return;

    setLoading(true);
    try {
      const formData = new FormData();
      formData.append('image', selectedImage);

      const response = await axios.post('/api/cartoonize', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        responseType: 'blob'
      });

      const resultUrl = URL.createObjectURL(response.data);
      setResultImage(resultUrl);
    } catch (error) {
      console.error('Error cartoonizing:', error);
      
      // Try to get error message from response
      let errorMessage = 'Failed to cartoonize image.';
      
      if (error.response) {
        // Backend returned an error
        if (error.response.data && error.response.data.error) {
          errorMessage = error.response.data.error;
        }
      } else if (!status.pix2pix) {
        errorMessage = '⚠️ Pix2Pix model is not loaded. Please train or download a model first.';
      }
      
      alert(errorMessage);
    } finally {
      setLoading(false);
    }
  };

  const handleGenerate = async () => {
    setLoading(true);
    try {
      const response = await axios.post('/api/generate', {
        num_images: numImages
      });

      setGeneratedImages(response.data.images);
    } catch (error) {
      console.error('Error generating:', error);
      
      // Try to get error message from response
      let errorMessage = 'Failed to generate images.';
      
      if (error.response) {
        // Backend returned an error
        if (error.response.data && error.response.data.error) {
          errorMessage = error.response.data.error;
        }
      } else if (!status.stylegan) {
        errorMessage = '⚠️ StyleGAN model is not loaded. Please train or download a model first.';
      }
      
      alert(errorMessage);
    } finally {
      setLoading(false);
    }
  };

  const downloadImage = (imageUrl, filename) => {
    const link = document.createElement('a');
    link.href = imageUrl;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="app">
      <header className="header">
        <h1>🎨 Cartoonist</h1>
        <p>AI-Powered Image Generation with GANs</p>
      </header>

      <div className="status-bar">
        <span className={`status-indicator ${status.pix2pix ? 'online' : 'offline'}`}>
          Pix2Pix: {status.pix2pix ? '●' : '○'}
        </span>
        <span className={`status-indicator ${status.stylegan ? 'online' : 'offline'}`}>
          StyleGAN: {status.stylegan ? '●' : '○'}
        </span>
      </div>

      <div className="tabs">
        <button 
          className={`tab ${activeTab === 'cartoonize' ? 'active' : ''}`}
          onClick={() => setActiveTab('cartoonize')}
        >
          Cartoonize Photo
        </button>
        <button 
          className={`tab ${activeTab === 'generate' ? 'active' : ''}`}
          onClick={() => setActiveTab('generate')}
        >
          Generate Faces
        </button>
      </div>

      <main className="content">
        {activeTab === 'cartoonize' && (
          <div className="cartoonize-section">
            <div className={`upload-area dropzone ${isDragActive ? 'active' : ''}`} {...getRootProps()}>
              <input {...getInputProps()} />
              {selectedImage ? (
                <div className="image-preview">
                  <img src={imagePreview || URL.createObjectURL(selectedImage)} alt="Selected" />
                  <p>{selectedImage.name}</p>
                </div>
              ) : (
                <div className="dropzone-content">
                  <p>📁 Drag & drop an image here, or click to select</p>
                  <p className="hint">Supports: JPG, PNG</p>
                </div>
              )}
            </div>

            <button 
              className="btn-primary"
              onClick={handleCartoonize}
              disabled={!selectedImage || loading || !status.pix2pix}
            >
              {loading ? 'Processing...' : '✨ Cartoonize'}
            </button>

            {resultImage && (
              <div className="result-section">
                <h3>Result</h3>
                <div className="comparison">
                  <div className="image-card">
                    <h4>Original</h4>
                    <img src={URL.createObjectURL(selectedImage)} alt="Original" />
                  </div>
                  <div className="arrow">→</div>
                  <div className="image-card">
                    <h4>Cartoon</h4>
                    <img src={resultImage} alt="Cartoon" />
                    <button 
                      className="btn-download"
                      onClick={() => downloadImage(resultImage, 'cartoon.png')}
                    >
                      Download
                    </button>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {activeTab === 'generate' && (
          <div className="generate-section">
            <div className="controls">
              <label>
                Number of images:
                <select value={numImages} onChange={(e) => setNumImages(Number(e.target.value))}>
                  {[4, 8, 16, 32].map(n => (
                    <option key={n} value={n}>{n}</option>
                  ))}
                </select>
              </label>
            </div>

            <button 
              className="btn-primary"
              onClick={handleGenerate}
              disabled={loading || !status.stylegan}
            >
              {loading ? 'Generating...' : '🎲 Generate Random Faces'}
            </button>

            {generatedImages.length > 0 && (
              <div className="gallery">
                <h3>Generated Faces ({generatedImages.length})</h3>
                <div className="gallery-grid">
                  {generatedImages.map((img, idx) => (
                    <div key={idx} className="gallery-item">
                      <img src={img} alt={`Generated ${idx + 1}`} />
                      <button 
                        className="btn-download-small"
                        onClick={() => downloadImage(img, `cartoon_${idx + 1}.png`)}
                      >
                        ⬇️
                      </button>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </main>

      <footer className="footer">
        <p>Powered by Pix2Pix & StyleGAN | Built with React</p>
      </footer>
    </div>
  );
}

export default App;

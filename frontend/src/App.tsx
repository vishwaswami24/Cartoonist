import { useEffect, useState } from 'react'
import { useDropzone } from 'react-dropzone'
import './App.css'
import {
  type BackendStatus,
  cartoonizeImage,
  checkStatus,
} from './api.ts'

type ConnectionState = 'checking' | 'online' | 'offline'

interface BusyState {
  active: boolean
  message: string
}

const defaultStatus: BackendStatus = {
  pix2pix: false,
  device: 'unavailable',
  cartoon_model: 'Unavailable',
}

function App() {
  const [status, setStatus] = useState<BackendStatus>(defaultStatus)
  const [connectionState, setConnectionState] = useState<ConnectionState>('checking')
  const [busyState, setBusyState] = useState<BusyState>({
    active: false,
    message: '',
  })
  const [notice, setNotice] = useState<string>('')
  const [selectedImage, setSelectedImage] = useState<File | null>(null)
  const [sourcePreview, setSourcePreview] = useState<string | null>(null)
  const [cartoonPreview, setCartoonPreview] = useState<string | null>(null)

  useEffect(() => {
    void refreshStatus()
  }, [])

  useEffect(() => {
    return () => {
      if (sourcePreview) {
        URL.revokeObjectURL(sourcePreview)
      }

      if (cartoonPreview) {
        URL.revokeObjectURL(cartoonPreview)
      }
    }
  }, [sourcePreview, cartoonPreview])

  async function refreshStatus() {
    setConnectionState('checking')

    try {
      const nextStatus = await checkStatus()
      setStatus(nextStatus)
      setConnectionState('online')
      setNotice('')
    } catch {
      setStatus(defaultStatus)
      setConnectionState('offline')
      setNotice('Backend unavailable. Start the Flask server to enable generation.')
    }
  }

  function replaceSourceImage(file: File | null) {
    setSelectedImage(file)
    setSourcePreview((currentPreview) => {
      if (currentPreview) {
        URL.revokeObjectURL(currentPreview)
      }

      return file ? URL.createObjectURL(file) : null
    })
  }

  function replaceCartoonPreview(nextPreview: string | null) {
    setCartoonPreview((currentPreview) => {
      if (currentPreview) {
        URL.revokeObjectURL(currentPreview)
      }

      return nextPreview
    })
  }

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    accept: {
      'image/png': ['.png'],
      'image/jpeg': ['.jpg', '.jpeg'],
      'image/webp': ['.webp'],
    },
    multiple: false,
    onDrop: (acceptedFiles) => {
      if (acceptedFiles.length === 0) {
        return
      }

      replaceSourceImage(acceptedFiles[0])
      replaceCartoonPreview(null)
      setNotice('')
    },
  })

  async function handleCartoonize() {
    if (!selectedImage) {
      setNotice('Choose a portrait or scene before cartoonizing.')
      return
    }

    setNotice('')
    setBusyState({
      active: true,
      message: 'Cartoonizing your image...',
    })

    try {
      const resultUrl = await cartoonizeImage(selectedImage)
      replaceCartoonPreview(resultUrl)
    } catch (error) {
      setNotice(error instanceof Error ? error.message : 'Unable to cartoonize this image.')
    } finally {
      setBusyState({
        active: false,
        message: '',
      })
    }
  }

  function downloadImage(imageUrl: string, filename: string) {
    const link = document.createElement('a')
    link.href = imageUrl
    link.download = filename
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
  }

  const backendLabel =
    connectionState === 'checking'
      ? 'Checking backend'
      : connectionState === 'online'
        ? 'Backend connected'
        : 'Backend offline'

  return (
    <div className="app-shell">
      <section className="hero-panel">
        <div className="hero-copy">
          <p className="eyebrow">Cartoonist Studio</p>
          <h1>Turn your photos into cartoons.</h1>
          <p className="hero-text">
            Upload a portrait, selfie, or scene and get back a stylized cartoon
            version in one focused workflow.
          </p>
          <div className="hero-facts">
            <span>One upload</span>
            <span>One render</span>
            <span>PNG download</span>
          </div>
          <div className="hero-actions">
            <button className="action-pill" onClick={() => document.getElementById('cartoon-workspace')?.scrollIntoView({ behavior: 'smooth' })}>
              Start Cartoonizing
            </button>
          </div>
        </div>

        <div className="status-grid">
          <article className={`status-card status-${connectionState}`}>
            <span className="status-label">System</span>
            <strong>{backendLabel}</strong>
            <p>{connectionState === 'online' ? `Running on ${status.device}` : 'Waiting for the API server.'}</p>
          </article>

          <article className={`status-card ${status.pix2pix ? 'ready' : 'idle'}`}>
            <span className="status-label">Cartoon Model</span>
            <strong>{status.pix2pix ? 'Ready to cartoonize' : 'Model unavailable'}</strong>
            <p>{status.cartoon_model ?? 'Upload one image and transform it into a stylized result.'}</p>
          </article>
          <article className="status-card status-tip">
            <span className="status-label">Best Input</span>
            <strong>Clear faces and simple lighting</strong>
            <p>Front-facing portraits and clean contrast usually give the strongest cartoon effect.</p>
          </article>
        </div>
      </section>

      <section className="workspace-panel" id="cartoon-workspace">
        <div className="workspace-header">
          <div>
            <p className="section-tag">Photo To Cartoon</p>
            <h2>Upload once, cartoonize, and download.</h2>
          </div>
          <button className="secondary-button" onClick={() => void refreshStatus()}>
            Refresh status
          </button>
        </div>

        {busyState.active ? <div className="banner banner-info">{busyState.message}</div> : null}
        {notice ? <div className="banner banner-warning">{notice}</div> : null}

        <div className="mode-grid">
          <div className="panel-card">
            <div className="card-heading">
              <div>
                <p className="section-tag">Upload</p>
                <h3>Drop a source image</h3>
              </div>
              <span className={`capability-badge ${status.pix2pix ? 'enabled' : 'disabled'}`}>
                {status.pix2pix ? 'Model ready' : 'Model needed'}
              </span>
            </div>

            <div className={`dropzone ${isDragActive ? 'active' : ''}`} {...getRootProps()}>
              <input {...getInputProps()} />
              {sourcePreview ? (
                <div className="preview-stack">
                  <img src={sourcePreview} alt="Selected upload preview" />
                  <p>{selectedImage?.name}</p>
                </div>
              ) : (
                <div className="dropzone-copy">
                  <p>Drag a portrait, selfie, or scene here.</p>
                  <span>PNG, JPG, or WEBP. Click if you prefer browsing.</span>
                </div>
              )}
            </div>

            <div className="action-row">
              <button
                className="primary-button"
                onClick={handleCartoonize}
                disabled={!status.pix2pix || !selectedImage || busyState.active}
              >
                Render cartoon
              </button>
              <button
                className="ghost-button"
                onClick={() => {
                  replaceSourceImage(null)
                  replaceCartoonPreview(null)
                  setNotice('')
                }}
              >
                Clear
              </button>
            </div>
          </div>

          <div className="panel-card">
            <div className="card-heading">
              <div>
                <p className="section-tag">Result</p>
                <h3>Before and after</h3>
              </div>
            </div>

            {sourcePreview && cartoonPreview ? (
              <div className="comparison-grid">
                <figure className="image-frame">
                  <img src={sourcePreview} alt="Original upload" />
                  <figcaption>Original</figcaption>
                </figure>
                <figure className="image-frame accent-frame">
                  <img src={cartoonPreview} alt="Cartoonized output" />
                  <figcaption>Cartoonized</figcaption>
                </figure>
              </div>
            ) : (
              <div className="empty-state">
                <p>Your rendered image will appear here once the cartoonizer finishes.</p>
              </div>
            )}

            {cartoonPreview ? (
              <button
                className="secondary-button"
                onClick={() => downloadImage(cartoonPreview, 'cartoonized-image.png')}
              >
                Download result
              </button>
            ) : null}
          </div>
        </div>
      </section>
    </div>
  )
}

export default App

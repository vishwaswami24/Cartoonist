import axios from 'axios'

export interface BackendStatus {
  pix2pix: boolean
  device: string
  cartoon_model?: string
}

const client = axios.create({
  baseURL: '/api',
  timeout: 90000,
})

async function extractErrorMessage(error: unknown, fallback: string) {
  if (!axios.isAxiosError(error)) {
    return fallback
  }

  const { data } = error.response ?? {}

  if (data instanceof Blob) {
    try {
      const parsed = JSON.parse(await data.text()) as { error?: string }
      return parsed.error ?? fallback
    } catch {
      return fallback
    }
  }

  if (typeof data === 'string') {
    try {
      const parsed = JSON.parse(data) as { error?: string }
      return parsed.error ?? fallback
    } catch {
      return data
    }
  }

  if (data && typeof data === 'object' && 'error' in data) {
    const apiError = data as { error?: string }
    return apiError.error ?? fallback
  }

  return error.message || fallback
}

export async function checkStatus() {
  try {
    const response = await client.get<BackendStatus>('/status')
    return response.data
  } catch (error) {
    throw new Error(await extractErrorMessage(error, 'Unable to reach the backend.'))
  }
}

export async function cartoonizeImage(imageFile: File) {
  const formData = new FormData()
  formData.append('image', imageFile)

  try {
    const response = await client.post<Blob>('/cartoonize', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      responseType: 'blob',
    })

    return URL.createObjectURL(response.data)
  } catch (error) {
    throw new Error(await extractErrorMessage(error, 'Cartoonization failed.'))
  }
}

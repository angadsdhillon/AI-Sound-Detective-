"use client"

import React, { useState, useCallback } from 'react'
import { Upload, FileAudio, X } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent } from '@/components/ui/card'
import { Progress } from '@/components/ui/progress'

interface FileUploadProps {
  onFileSelected: (file: File) => void
  onUploadComplete: (fileId: string, filename: string) => void
  isUploading: boolean
}

export default function FileUpload({ onFileSelected, onUploadComplete, isUploading }: FileUploadProps) {
  const [dragActive, setDragActive] = useState(false)
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [uploadProgress, setUploadProgress] = useState(0)

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true)
    } else if (e.type === "dragleave") {
      setDragActive(false)
    }
  }, [])

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0])
    }
  }, [])

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    e.preventDefault()
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0])
    }
  }

  const handleFile = (file: File) => {
    // Validate file type
    const allowedTypes = [
      'audio/mpeg', 'audio/wav', 'audio/x-wav', 'audio/mp4', 'audio/x-m4a',
      'audio/flac', 'audio/ogg', 'video/mp4', 'video/quicktime', 'video/x-msvideo'
    ]
    
    const allowedExtensions = ['.mp3', '.wav', '.m4a', '.flac', '.ogg', '.mp4', '.mov', '.avi', '.mkv']
    const fileExtension = '.' + file.name.split('.').pop()?.toLowerCase()

    if (!allowedTypes.includes(file.type) && !allowedExtensions.includes(fileExtension)) {
      alert('Please upload an audio or video file (mp3, wav, m4a, mp4, mov, etc.)')
      return
    }

    // Validate file size (100MB)
    if (file.size > 100 * 1024 * 1024) {
      alert('File size must be less than 100MB')
      return
    }

    setSelectedFile(file)
    onFileSelected(file)
  }

  const clearFile = () => {
    setSelectedFile(null)
    setUploadProgress(0)
  }

  const uploadFile = async () => {
    if (!selectedFile) return

    const formData = new FormData()
    formData.append('file', selectedFile)

    try {
      // Simulate upload progress
      const progressInterval = setInterval(() => {
        setUploadProgress(prev => {
          if (prev >= 90) {
            clearInterval(progressInterval)
            return 90
          }
          return prev + 10
        })
      }, 200)

      const response = await fetch('http://localhost:8000/api/v1/upload', {
        method: 'POST',
        body: formData,
      })

      clearInterval(progressInterval)
      setUploadProgress(100)

      if (!response.ok) {
        throw new Error('Upload failed')
      }

      const data = await response.json()
      onUploadComplete(data.file_id, data.filename)
      
    } catch (error) {
      console.error('Upload error:', error)
      alert('Upload failed. Please try again.')
      setUploadProgress(0)
    }
  }

  return (
    <Card className="w-full">
      <CardContent className="pt-6">
        <div
          className={`relative border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
            dragActive ? 'border-primary bg-primary/5' : 'border-gray-300'
          }`}
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
        >
          <input
            type="file"
            id="file-upload"
            className="hidden"
            onChange={handleChange}
            accept=".mp3,.wav,.m4a,.flac,.ogg,.mp4,.mov,.avi,.mkv,audio/*,video/*"
          />
          
          {!selectedFile ? (
            <label htmlFor="file-upload" className="cursor-pointer">
              <Upload className="mx-auto h-12 w-12 text-gray-400 mb-4" />
              <p className="text-lg font-medium text-gray-700 mb-2">
                Drop your audio or video file here
              </p>
              <p className="text-sm text-gray-500 mb-4">
                or click to browse
              </p>
              <p className="text-xs text-gray-400">
                Supports: MP3, WAV, M4A, FLAC, MP4, MOV (max 100MB)
              </p>
            </label>
          ) : (
            <div className="space-y-4">
              <div className="flex items-center justify-center gap-3">
                <FileAudio className="h-8 w-8 text-primary" />
                <div className="text-left">
                  <p className="font-medium text-gray-900">{selectedFile.name}</p>
                  <p className="text-sm text-gray-500">
                    {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
                  </p>
                </div>
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={clearFile}
                  disabled={isUploading}
                >
                  <X className="h-4 w-4" />
                </Button>
              </div>

              {uploadProgress > 0 && uploadProgress < 100 && (
                <Progress value={uploadProgress} className="w-full" />
              )}

              {uploadProgress === 0 && (
                <Button onClick={uploadFile} disabled={isUploading} className="w-full">
                  Upload & Analyze
                </Button>
              )}
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  )
}

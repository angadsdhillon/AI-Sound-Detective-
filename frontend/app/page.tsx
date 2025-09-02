"use client"

import React, { useState } from 'react'
import FileUpload from '@/components/FileUpload'
import AudioRecorder from '@/components/AudioRecorder'
import SoundReportDashboard from '@/components/SoundReportDashboard'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Card, CardContent } from '@/components/ui/card'
import { Loader2 } from 'lucide-react'

export default function Home() {
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [analysisResult, setAnalysisResult] = useState<any>(null)
  const [currentFileId, setCurrentFileId] = useState<string | null>(null)

  const handleFileUpload = async (fileId: string, filename: string) => {
    setCurrentFileId(fileId)
    setIsAnalyzing(true)
    
    try {
      const response = await fetch(
        `http://localhost:8000/api/v1/analyze/${fileId}?use_ensemble=true&detect_anomalies=true&find_similar=true`,
        {
          method: 'POST',
        }
      )

      if (!response.ok) {
        throw new Error('Analysis failed')
      }

      const result = await response.json()
      setAnalysisResult(result)
    } catch (error) {
      console.error('Analysis error:', error)
      alert('Analysis failed. Please make sure the backend is running.')
    } finally {
      setIsAnalyzing(false)
    }
  }

  const handleRecordingComplete = async (audioBlob: Blob) => {
    setIsAnalyzing(true)

    try {
      // Upload recorded audio
      const formData = new FormData()
      formData.append('file', audioBlob, 'recording.wav')

      const uploadResponse = await fetch('http://localhost:8000/api/v1/upload', {
        method: 'POST',
        body: formData,
      })

      if (!uploadResponse.ok) {
        throw new Error('Upload failed')
      }

      const uploadData = await uploadResponse.json()
      
      // Analyze
      await handleFileUpload(uploadData.file_id, uploadData.filename)
    } catch (error) {
      console.error('Recording analysis error:', error)
      alert('Failed to analyze recording. Please ensure the backend is running.')
      setIsAnalyzing(false)
    }
  }

  const handleExportPDF = async () => {
    if (!currentFileId) return

    try {
      const response = await fetch(
        `http://localhost:8000/api/v1/report/${currentFileId}`,
        {
          method: 'GET',
        }
      )

      if (!response.ok) {
        throw new Error('PDF generation failed')
      }

      const blob = await response.blob()
      const url = window.URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `sound-report-${currentFileId}.pdf`
      document.body.appendChild(a)
      a.click()
      window.URL.revokeObjectURL(url)
      document.body.removeChild(a)
    } catch (error) {
      console.error('PDF export error:', error)
      alert('PDF export is not yet implemented in the backend.')
    }
  }

  const handleReset = () => {
    setAnalysisResult(null)
    setCurrentFileId(null)
  }

  return (
    <main className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="text-center mb-12">
          <h1 className="text-5xl font-bold mb-4 bg-clip-text text-transparent bg-gradient-to-r from-blue-600 to-purple-600">
            AI Sound Detective
          </h1>
          <p className="text-xl text-gray-600 mb-2">
            Advanced ML-Powered Sound Analysis Platform
          </p>
          <p className="text-sm text-gray-500">
            Upload audio or record sounds to unlock deep AI insights
          </p>
        </div>

        {!analysisResult ? (
          <div className="max-w-4xl mx-auto">
            {isAnalyzing ? (
              <Card>
                <CardContent className="pt-12 pb-12">
                  <div className="flex flex-col items-center justify-center space-y-4">
                    <Loader2 className="h-16 w-16 animate-spin text-primary" />
                    <h3 className="text-xl font-semibold">Analyzing Sound...</h3>
                    <p className="text-gray-500 text-center max-w-md">
                      Running ML models: Classification, Anomaly Detection, Feature Extraction, and Similarity Search
                    </p>
                  </div>
                </CardContent>
              </Card>
            ) : (
              <Tabs defaultValue="upload" className="w-full">
                <TabsList className="grid w-full grid-cols-2 mb-8">
                  <TabsTrigger value="upload" className="text-lg">
                    Upload File
                  </TabsTrigger>
                  <TabsTrigger value="record" className="text-lg">
                    Record Audio
                  </TabsTrigger>
                </TabsList>

                <TabsContent value="upload">
                  <FileUpload
                    onFileSelected={() => {}}
                    onUploadComplete={handleFileUpload}
                    isUploading={isAnalyzing}
                  />
                </TabsContent>

                <TabsContent value="record">
                  <AudioRecorder onRecordingComplete={handleRecordingComplete} />
                </TabsContent>
              </Tabs>
            )}

            {/* Features Section */}
            {!isAnalyzing && (
              <div className="mt-16 grid md:grid-cols-2 lg:grid-cols-4 gap-6">
                <Card className="text-center">
                  <CardContent className="pt-6">
                    <div className="text-4xl mb-2">🎵</div>
                    <h3 className="font-semibold mb-2">Sound Classification</h3>
                    <p className="text-sm text-gray-600">
                      Deep learning models identify sound categories
                    </p>
                  </CardContent>
                </Card>

                <Card className="text-center">
                  <CardContent className="pt-6">
                    <div className="text-4xl mb-2">📊</div>
                    <h3 className="font-semibold mb-2">Spectrogram Analysis</h3>
                    <p className="text-sm text-gray-600">
                      Visual frequency decomposition over time
                    </p>
                  </CardContent>
                </Card>

                <Card className="text-center">
                  <CardContent className="pt-6">
                    <div className="text-4xl mb-2">🔍</div>
                    <h3 className="font-semibold mb-2">Anomaly Detection</h3>
                    <p className="text-sm text-gray-600">
                      Identifies unusual patterns and artifacts
                    </p>
                  </CardContent>
                </Card>

                <Card className="text-center">
                  <CardContent className="pt-6">
                    <div className="text-4xl mb-2">🎯</div>
                    <h3 className="font-semibold mb-2">Similarity Search</h3>
                    <p className="text-sm text-gray-600">
                      Find similar sounds using fingerprinting
                    </p>
                  </CardContent>
                </Card>
              </div>
            )}
          </div>
        ) : (
          <div>
            <div className="mb-6">
              <button
                onClick={handleReset}
                className="text-blue-600 hover:text-blue-800 font-medium"
              >
                ← Analyze Another Sound
              </button>
            </div>
            <SoundReportDashboard result={analysisResult} onExportPDF={handleExportPDF} />
          </div>
        )}

        {/* Footer */}
        <footer className="mt-16 text-center text-gray-500 text-sm">
          <p>
            AI Sound Detective • Powered by PyTorch, FastAPI, and Next.js
          </p>
        </footer>
      </div>
    </main>
  )
}

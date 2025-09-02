"use client"

import React from 'react'
import dynamic from 'next/dynamic'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Progress } from '@/components/ui/progress'
import { AlertCircle, CheckCircle, Download, Music } from 'lucide-react'
import { Button } from '@/components/ui/button'

// Dynamic import for Plotly to avoid SSR issues
const Plot = dynamic(() => import('react-plotly.js'), { ssr: false })

interface AnalysisResult {
  id: string
  filename: string
  timestamp: string
  metadata: {
    duration: number
    sample_rate: number
    samples: number
    rms_energy: number
    max_amplitude: number
    zero_crossing_rate: number
  }
  classification: {
    method: string
    top_class: string
    top_probability: number
    all_predictions: Array<{
      class: string
      probability: number
      confidence: number
    }>
  }
  anomaly_detection?: {
    is_anomalous: boolean
    anomaly_score: number
    anomalies_detected: Array<{
      type: string
      severity: string
      score: number
      description: string
    }>
  }
  similar_sounds?: Array<{
    id: string
    name: string
    similarity: number
    category: string
  }>
  spectrogram?: {
    frequencies: number[]
    times: number[]
    magnitude: number[][]
  }
  fft_3d?: {
    frequencies: number[]
    times: number[]
    magnitude: number[][]
    magnitude_db: number[][]
  }
  waveform?: number[]
}

interface SoundReportDashboardProps {
  result: AnalysisResult
  onExportPDF: () => void
}

export default function SoundReportDashboard({ result, onExportPDF }: SoundReportDashboardProps) {
  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'high': return 'text-red-600 bg-red-50'
      case 'medium': return 'text-yellow-600 bg-yellow-50'
      case 'low': return 'text-blue-600 bg-blue-50'
      default: return 'text-gray-600 bg-gray-50'
    }
  }

  return (
    <div className="space-y-6 w-full">
      {/* Header */}
      <div className="flex justify-between items-start">
        <div>
          <h1 className="text-3xl font-bold mb-2">Sound Analysis Report</h1>
          <p className="text-gray-500">
            File: {result.filename} • Duration: {result.metadata.duration.toFixed(2)}s
          </p>
        </div>
        <Button onClick={onExportPDF} className="gap-2">
          <Download className="h-4 w-4" />
          Export PDF
        </Button>
      </div>

      {/* Metadata Summary */}
      <Card>
        <CardHeader>
          <CardTitle>Audio Metadata</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
            <div>
              <p className="text-sm text-gray-500">Sample Rate</p>
              <p className="text-lg font-semibold">{result.metadata.sample_rate} Hz</p>
            </div>
            <div>
              <p className="text-sm text-gray-500">Samples</p>
              <p className="text-lg font-semibold">{result.metadata.samples.toLocaleString()}</p>
            </div>
            <div>
              <p className="text-sm text-gray-500">RMS Energy</p>
              <p className="text-lg font-semibold">{result.metadata.rms_energy.toFixed(4)}</p>
            </div>
            <div>
              <p className="text-sm text-gray-500">Max Amplitude</p>
              <p className="text-lg font-semibold">{result.metadata.max_amplitude.toFixed(4)}</p>
            </div>
            <div>
              <p className="text-sm text-gray-500">Zero Crossing Rate</p>
              <p className="text-lg font-semibold">{result.metadata.zero_crossing_rate.toFixed(4)}</p>
            </div>
            <div>
              <p className="text-sm text-gray-500">Duration</p>
              <p className="text-lg font-semibold">{result.metadata.duration.toFixed(2)}s</p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Classification Results */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Music className="h-5 w-5" />
            Sound Classification
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="bg-primary/5 p-4 rounded-lg">
            <p className="text-sm text-gray-600 mb-1">Top Prediction</p>
            <p className="text-2xl font-bold text-primary">{result.classification.top_class}</p>
            <p className="text-lg text-gray-600">
              {(result.classification.top_probability * 100).toFixed(1)}% confidence
            </p>
          </div>

          <div className="space-y-2">
            <p className="font-semibold text-sm text-gray-700">All Predictions:</p>
            {result.classification.all_predictions.slice(0, 5).map((pred, idx) => (
              <div key={idx} className="space-y-1">
                <div className="flex justify-between text-sm">
                  <span className="font-medium">{pred.class}</span>
                  <span className="text-gray-600">{pred.confidence.toFixed(1)}%</span>
                </div>
                <Progress value={pred.confidence} className="h-2" />
              </div>
            ))}
          </div>

          <p className="text-xs text-gray-500 mt-4">
            Method: {result.classification.method}
          </p>
        </CardContent>
      </Card>

      {/* Anomaly Detection */}
      {result.anomaly_detection && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              {result.anomaly_detection.is_anomalous ? (
                <AlertCircle className="h-5 w-5 text-red-500" />
              ) : (
                <CheckCircle className="h-5 w-5 text-green-500" />
              )}
              Anomaly Detection
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className={`p-4 rounded-lg ${
              result.anomaly_detection.is_anomalous 
                ? 'bg-red-50 border border-red-200' 
                : 'bg-green-50 border border-green-200'
            }`}>
              <p className="font-semibold mb-1">
                {result.anomaly_detection.is_anomalous 
                  ? '⚠️ Anomalies Detected' 
                  : '✓ No Anomalies Detected'}
              </p>
              <p className="text-sm text-gray-600">
                Anomaly Score: {result.anomaly_detection.anomaly_score.toFixed(4)}
              </p>
            </div>

            {result.anomaly_detection.anomalies_detected.length > 0 && (
              <div className="space-y-2">
                <p className="font-semibold text-sm">Detected Issues:</p>
                {result.anomaly_detection.anomalies_detected.map((anomaly, idx) => (
                  <div 
                    key={idx} 
                    className={`p-3 rounded-lg ${getSeverityColor(anomaly.severity)}`}
                  >
                    <div className="flex justify-between items-start mb-1">
                      <span className="font-semibold text-sm">{anomaly.type}</span>
                      <span className="text-xs uppercase">{anomaly.severity}</span>
                    </div>
                    <p className="text-sm">{anomaly.description}</p>
                    <p className="text-xs mt-1">Score: {anomaly.score.toFixed(4)}</p>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* Similar Sounds */}
      {result.similar_sounds && result.similar_sounds.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Similar Sounds</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {result.similar_sounds.map((sound, idx) => (
                <div key={idx} className="flex justify-between items-center p-3 bg-gray-50 rounded-lg">
                  <div>
                    <p className="font-medium">{sound.name}</p>
                    <p className="text-sm text-gray-500">{sound.category}</p>
                  </div>
                  <div className="text-right">
                    <p className="font-semibold text-primary">
                      {(sound.similarity * 100).toFixed(1)}%
                    </p>
                    <p className="text-xs text-gray-500">similarity</p>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Visualizations */}
      <Card>
        <CardHeader>
          <CardTitle>Visualizations</CardTitle>
        </CardHeader>
        <CardContent>
          <Tabs defaultValue="waveform" className="w-full">
            <TabsList className="grid w-full grid-cols-3">
              <TabsTrigger value="waveform">Waveform</TabsTrigger>
              <TabsTrigger value="spectrogram">Spectrogram</TabsTrigger>
              <TabsTrigger value="3d-fft">3D FFT</TabsTrigger>
            </TabsList>

            <TabsContent value="waveform" className="mt-4">
              {result.waveform && (
                <Plot
                  data={[
                    {
                      y: result.waveform,
                      type: 'scatter',
                      mode: 'lines',
                      line: { color: '#3b82f6', width: 1 },
                      name: 'Amplitude'
                    }
                  ]}
                  layout={{
                    title: 'Audio Waveform',
                    xaxis: { title: 'Sample' },
                    yaxis: { title: 'Amplitude' },
                    autosize: true,
                    height: 400
                  }}
                  useResizeHandler={true}
                  style={{ width: '100%' }}
                  config={{ responsive: true }}
                />
              )}
            </TabsContent>

            <TabsContent value="spectrogram" className="mt-4">
              {result.spectrogram && (
                <Plot
                  data={[
                    {
                      z: result.spectrogram.magnitude,
                      x: result.spectrogram.times,
                      y: result.spectrogram.frequencies,
                      type: 'heatmap',
                      colorscale: 'Viridis',
                      colorbar: { title: 'Magnitude (dB)' }
                    }
                  ]}
                  layout={{
                    title: 'Mel Spectrogram',
                    xaxis: { title: 'Time (s)' },
                    yaxis: { title: 'Frequency (Hz)' },
                    autosize: true,
                    height: 400
                  }}
                  useResizeHandler={true}
                  style={{ width: '100%' }}
                  config={{ responsive: true }}
                />
              )}
            </TabsContent>

            <TabsContent value="3d-fft" className="mt-4">
              {result.fft_3d && (
                <Plot
                  data={[
                    {
                      z: result.fft_3d.magnitude_db,
                      x: result.fft_3d.times,
                      y: result.fft_3d.frequencies,
                      type: 'surface',
                      colorscale: 'Jet',
                      colorbar: { title: 'Magnitude (dB)' }
                    }
                  ]}
                  layout={{
                    title: '3D Frequency Spectrum',
                    scene: {
                      xaxis: { title: 'Time (s)' },
                      yaxis: { title: 'Frequency (Hz)' },
                      zaxis: { title: 'Magnitude (dB)' }
                    },
                    autosize: true,
                    height: 500
                  }}
                  useResizeHandler={true}
                  style={{ width: '100%' }}
                  config={{ responsive: true }}
                />
              )}
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>
    </div>
  )
}

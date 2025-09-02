"use client"

import React, { useState, useRef, useEffect } from 'react'
import { Mic, Square, Play, Pause, Trash2 } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'

interface AudioRecorderProps {
  onRecordingComplete: (audioBlob: Blob) => void
}

export default function AudioRecorder({ onRecordingComplete }: AudioRecorderProps) {
  const [isRecording, setIsRecording] = useState(false)
  const [isPaused, setIsPaused] = useState(false)
  const [recordingTime, setRecordingTime] = useState(0)
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null)
  const [audioURL, setAudioURL] = useState<string>('')
  const [waveformData, setWaveformData] = useState<number[]>([])

  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const audioChunksRef = useRef<Blob[]>([])
  const timerRef = useRef<NodeJS.Timeout | null>(null)
  const audioContextRef = useRef<AudioContext | null>(null)
  const analyserRef = useRef<AnalyserNode | null>(null)
  const animationRef = useRef<number | null>(null)

  useEffect(() => {
    return () => {
      if (timerRef.current) clearInterval(timerRef.current)
      if (animationRef.current) cancelAnimationFrame(animationRef.current)
      if (audioContextRef.current) audioContextRef.current.close()
    }
  }, [])

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      
      // Setup audio context for visualization
      audioContextRef.current = new AudioContext()
      const source = audioContextRef.current.createMediaStreamSource(stream)
      analyserRef.current = audioContextRef.current.createAnalyser()
      analyserRef.current.fftSize = 256
      source.connect(analyserRef.current)
      
      // Setup media recorder
      mediaRecorderRef.current = new MediaRecorder(stream)
      audioChunksRef.current = []

      mediaRecorderRef.current.ondataavailable = (event) => {
        audioChunksRef.current.push(event.data)
      }

      mediaRecorderRef.current.onstop = () => {
        const blob = new Blob(audioChunksRef.current, { type: 'audio/wav' })
        setAudioBlob(blob)
        setAudioURL(URL.createObjectURL(blob))
        stream.getTracks().forEach(track => track.stop())
        if (audioContextRef.current) audioContextRef.current.close()
      }

      mediaRecorderRef.current.start()
      setIsRecording(true)
      setIsPaused(false)

      // Start timer
      timerRef.current = setInterval(() => {
        setRecordingTime(prev => prev + 1)
      }, 1000)

      // Start visualization
      visualize()

    } catch (error) {
      console.error('Error accessing microphone:', error)
      alert('Could not access microphone. Please grant permission.')
    }
  }

  const visualize = () => {
    if (!analyserRef.current) return

    const bufferLength = analyserRef.current.frequencyBinCount
    const dataArray = new Uint8Array(bufferLength)

    const draw = () => {
      if (!analyserRef.current || !isRecording) return

      animationRef.current = requestAnimationFrame(draw)
      analyserRef.current.getByteTimeDomainData(dataArray)

      // Downsample for display
      const samples = 50
      const step = Math.floor(bufferLength / samples)
      const waveform: number[] = []
      for (let i = 0; i < samples; i++) {
        waveform.push(dataArray[i * step])
      }
      setWaveformData(waveform)
    }

    draw()
  }

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop()
      setIsRecording(false)
      setIsPaused(false)
      if (timerRef.current) clearInterval(timerRef.current)
      if (animationRef.current) cancelAnimationFrame(animationRef.current)
    }
  }

  const pauseRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      if (isPaused) {
        mediaRecorderRef.current.resume()
        timerRef.current = setInterval(() => {
          setRecordingTime(prev => prev + 1)
        }, 1000)
      } else {
        mediaRecorderRef.current.pause()
        if (timerRef.current) clearInterval(timerRef.current)
      }
      setIsPaused(!isPaused)
    }
  }

  const deleteRecording = () => {
    setAudioBlob(null)
    setAudioURL('')
    setRecordingTime(0)
    setWaveformData([])
    if (audioURL) URL.revokeObjectURL(audioURL)
  }

  const handleAnalyze = () => {
    if (audioBlob) {
      onRecordingComplete(audioBlob)
    }
  }

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60)
    const secs = seconds % 60
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`
  }

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle>Record Audio</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Waveform Visualization */}
        <div className="h-24 bg-gray-100 rounded-lg flex items-center justify-center overflow-hidden">
          {isRecording ? (
            <div className="flex items-center justify-center h-full w-full gap-1 px-4">
              {waveformData.map((value, index) => {
                const height = ((value - 128) / 128) * 100
                return (
                  <div
                    key={index}
                    className="bg-primary transition-all duration-100"
                    style={{
                      width: '2%',
                      height: `${Math.abs(height)}%`,
                      minHeight: '2px'
                    }}
                  />
                )
              })}
            </div>
          ) : audioURL ? (
            <audio src={audioURL} controls className="w-full" />
          ) : (
            <p className="text-gray-400">Waveform will appear here</p>
          )}
        </div>

        {/* Timer */}
        {(isRecording || recordingTime > 0) && (
          <div className="text-center">
            <p className="text-2xl font-mono font-bold text-primary">
              {formatTime(recordingTime)}
            </p>
          </div>
        )}

        {/* Controls */}
        <div className="flex gap-2 justify-center">
          {!isRecording && !audioBlob && (
            <Button onClick={startRecording} size="lg" className="gap-2">
              <Mic className="h-5 w-5" />
              Start Recording
            </Button>
          )}

          {isRecording && (
            <>
              <Button onClick={pauseRecording} variant="outline" size="lg" className="gap-2">
                {isPaused ? <Play className="h-5 w-5" /> : <Pause className="h-5 w-5" />}
                {isPaused ? 'Resume' : 'Pause'}
              </Button>
              <Button onClick={stopRecording} variant="destructive" size="lg" className="gap-2">
                <Square className="h-5 w-5" />
                Stop
              </Button>
            </>
          )}

          {audioBlob && !isRecording && (
            <>
              <Button onClick={handleAnalyze} size="lg" className="gap-2">
                Analyze Recording
              </Button>
              <Button onClick={deleteRecording} variant="outline" size="lg" className="gap-2">
                <Trash2 className="h-5 w-5" />
                Delete
              </Button>
            </>
          )}
        </div>
      </CardContent>
    </Card>
  )
}

import React, { useState, useCallback } from 'react';
import { Brain, AlertCircle, Zap } from 'lucide-react';
import { ApiTokenInput } from './components/ApiTokenInput';
import { TextInput } from './components/TextInput';
import { ResultsDisplay } from './components/ResultsDisplay';
import { SentimentVisualization } from './components/SentimentVisualization';
import { SentimentAnalysisService } from './services/sentimentApi';
import { SentimentResult, BatchAnalysisResult } from './types/sentiment';
import { exportToCSV, exportToJSON, exportToPDF } from './utils/exportUtils';

function App() {
  const [results, setResults] = useState<SentimentResult[]>([]);
  const [batchResults, setBatchResults] = useState<BatchAnalysisResult[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [hasApiToken, setHasApiToken] = useState(false);
  const [showTokenInput, setShowTokenInput] = useState(true);

  const sentimentService = SentimentAnalysisService.getInstance();

  const processSentimentResponse = (apiResponse: any[], text: string): SentimentResult => {
    // Map Hugging Face labels to readable sentiment
    const labelMap: Record<string, string> = {
      'LABEL_0': 'negative',
      'LABEL_1': 'neutral', 
      'LABEL_2': 'positive'
    };

    const topResult = apiResponse[0];
    const sentiment = labelMap[topResult.label] || 'neutral';
    const confidence = topResult.score;

    return {
      id: Date.now().toString() + Math.random().toString(36).substr(2, 9),
      text,
      sentiment: sentiment as 'positive' | 'negative' | 'neutral',
      confidence,
      keywords: sentimentService.extractKeywords(text, sentiment),
      timestamp: new Date(),
    };
  };

  const handleSingleAnalysis = useCallback(async (text: string) => {
    setIsLoading(true);
    setError(null);

    try {
      const apiResponse = await sentimentService.analyzeSentiment(text);
      const result = processSentimentResponse(apiResponse, text);
      setResults(prev => [result, ...prev]);
    } catch (err) {
      setError('Failed to analyze sentiment. Please try again.');
      console.error('Analysis error:', err);
    } finally {
      setIsLoading(false);
    }
  }, [sentimentService]);

  const handleBatchAnalysis = useCallback(async (texts: string[]) => {
    setIsLoading(true);
    setError(null);

    try {
      const apiResponses = await sentimentService.analyzeBatch(texts);
      const batchResultsData = texts.map((text, index) => 
        processSentimentResponse(apiResponses[index], text)
      );
      
      // Create batch analysis summary
      const batchSummary: BatchAnalysisResult = {
        id: Date.now().toString(),
        results: batchResultsData,
        summary: {
          total: batchResultsData.length,
          positive: batchResultsData.filter(r => r.sentiment === 'positive').length,
          negative: batchResultsData.filter(r => r.sentiment === 'negative').length,
          neutral: batchResultsData.filter(r => r.sentiment === 'neutral').length,
          averageConfidence: batchResultsData.reduce((sum, r) => sum + r.confidence, 0) / batchResultsData.length,
        },
        timestamp: new Date(),
      };

      setResults(prev => [...batchResultsData, ...prev]);
      setBatchResults(prev => [batchSummary, ...prev]);
    } catch (err) {
      setError('Failed to analyze batch. Please try again.');
      console.error('Batch analysis error:', err);
    } finally {
      setIsLoading(false);
    }
  }, [sentimentService]);

  const handleExport = useCallback((format: 'csv' | 'json' | 'pdf') => {
    if (results.length === 0) return;

    try {
      switch (format) {
        case 'csv':
          exportToCSV(results);
          break;
        case 'json':
          exportToJSON(results);
          break;
        case 'pdf':
          exportToPDF(results);
          break;
      }
    } catch (err) {
      setError(`Failed to export ${format.toUpperCase()}. Please try again.`);
      console.error('Export error:', err);
    }
  }, [results]);

  const handleTokenSet = (hasToken: boolean) => {
    setHasApiToken(hasToken);
    setShowTokenInput(false);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="flex items-center justify-center gap-3 mb-4">
            <div className="p-3 bg-blue-100 rounded-xl">
              <Brain className="text-blue-600" size={32} />
            </div>
            <h1 className="text-4xl font-bold text-gray-900">Enhanced Sentiment Analysis Dashboard</h1>
          </div>
          <p className="text-gray-600 text-lg max-w-3xl mx-auto">
            Advanced emotional tone analysis with enhanced accuracy using ensemble AI models, 
            sophisticated preprocessing, and context-aware sentiment detection.
          </p>
          
          {/* Enhanced Features Banner */}
          <div className="mt-6 bg-gradient-to-r from-green-50 to-blue-50 border border-green-200 rounded-lg p-4 max-w-4xl mx-auto">
            <div className="flex items-center justify-center gap-2 mb-2">
              <Zap className="text-green-600" size={20} />
              <span className="font-semibold text-green-800">Enhanced Accuracy Features</span>
            </div>
            <div className="grid md:grid-cols-3 gap-4 text-sm text-green-700">
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                <span>Advanced Text Preprocessing</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                <span>Ensemble AI Models</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                <span>Context-Aware Analysis</span>
              </div>
            </div>
          </div>
          
          {!hasApiToken && (
            <div className="mt-4 p-3 bg-purple-50 border border-purple-200 rounded-lg inline-block">
              <div className="flex items-center gap-2 text-purple-800">
                <AlertCircle size={16} />
                <span className="text-sm">Running in enhanced mode with advanced accuracy features</span>
              </div>
            </div>
          )}
        </div>

        {/* API Token Input */}
        {showTokenInput && (
          <div className="max-w-2xl mx-auto mb-8">
            <ApiTokenInput onTokenSet={handleTokenSet} />
          </div>
        )}

        {!showTokenInput && (
          <div className="space-y-8">
            {/* Error Display */}
            {error && (
              <div className="bg-red-50 border border-red-200 rounded-lg p-4">
                <div className="flex items-center gap-2 text-red-800">
                  <AlertCircle size={16} />
                  <span>{error}</span>
                </div>
              </div>
            )}

            {/* Main Content Grid */}
            <div className="grid lg:grid-cols-2 gap-8">
              {/* Left Column - Input */}
              <div className="space-y-6">
                <TextInput
                  onAnalyze={handleSingleAnalysis}
                  onBatchAnalyze={handleBatchAnalysis}
                  isLoading={isLoading}
                />
                
                {!hasApiToken && (
                  <div className="text-center">
                    <button
                      onClick={() => setShowTokenInput(true)}
                      className="text-blue-600 hover:text-blue-700 text-sm underline"
                    >
                      Configure API Token for Maximum Accuracy
                    </button>
                  </div>
                )}
              </div>

              {/* Right Column - Results */}
              <div>
                <ResultsDisplay
                  results={results}
                  onExport={handleExport}
                />
              </div>
            </div>

            {/* Visualization Section */}
            {results.length > 0 && (
              <div className="mt-8">
                <SentimentVisualization
                  results={results}
                  batchResults={batchResults}
                />
              </div>
            )}

            {/* Enhanced Features Footer */}
            <div className="bg-green-50 rounded-xl shadow-lg p-8 border border-green-200 mt-12">
              <h3 className="text-xl font-semibold text-green-800 mb-4 text-center">Enhanced Analysis Capabilities</h3>
              <div className="grid md:grid-cols-3 gap-6">
                <div className="text-center">
                  <div className="p-3 bg-blue-100 rounded-lg inline-block mb-3">
                    <Brain className="text-blue-600" size={24} />
                  </div>
                  <h4 className="font-medium text-gray-900 mb-2">Ensemble AI Models</h4>
                  <p className="text-sm text-gray-600">
                    Multiple transformer models working together with weighted averaging for superior accuracy
                  </p>
                </div>
                <div className="text-center">
                  <div className="p-3 bg-green-100 rounded-lg inline-block mb-3">
                    <Zap className="text-green-600" size={24} />
                  </div>
                  <h4 className="font-medium text-gray-900 mb-2">Advanced Preprocessing</h4>
                  <p className="text-sm text-gray-600">
                    Sophisticated text processing with negation handling, emoji recognition, and context analysis
                  </p>
                </div>
                <div className="text-center">
                  <div className="p-3 bg-purple-100 rounded-lg inline-block mb-3">
                    <svg className="w-6 h-6 text-purple-600" fill="currentColor" viewBox="0 0 20 20">
                      <path d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                  </div>
                  <h4 className="font-medium text-gray-900 mb-2">Context-Aware Analysis</h4>
                  <p className="text-sm text-gray-600">
                    Intelligent understanding of sentence structure, intensifiers, and linguistic patterns
                  </p>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;

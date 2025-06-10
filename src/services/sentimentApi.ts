import axios from 'axios';
import { ApiResponse } from '../types/sentiment';

// Multiple model endpoints for ensemble analysis
const SENTIMENT_MODELS = {
  primary: 'https://api-inference.huggingface.co/models/cardiffnlp/twitter-roberta-base-sentiment-latest',
  secondary: 'https://api-inference.huggingface.co/models/nlptown/bert-base-multilingual-uncased-sentiment',
  tertiary: 'https://api-inference.huggingface.co/models/j-hartmann/emotion-english-distilroberta-base'
};

export class SentimentAnalysisService {
  private static instance: SentimentAnalysisService;
  private apiToken: string;
  private useEnsemble: boolean = true;

  private constructor() {
    this.apiToken = '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';
  }

  public static getInstance(): SentimentAnalysisService {
    if (!SentimentAnalysisService.instance) {
      SentimentAnalysisService.instance = new SentimentAnalysisService();
    }
    return SentimentAnalysisService.instance;
  }

  public setApiToken(token: string): void {
    this.apiToken = token;
  }

  private preprocessText(text: string): string {
    // Enhanced text preprocessing for better accuracy
    let processed = text.trim();
    
    // Handle negations more effectively
    processed = processed.replace(/\b(not|no|never|nothing|nowhere|nobody|none|neither|nor)\s+/gi, 'NOT_');
    
    // Handle intensifiers
    processed = processed.replace(/\b(very|extremely|incredibly|absolutely|totally|completely)\s+/gi, 'INTENSIFIER_');
    
    // Handle diminishers
    processed = processed.replace(/\b(slightly|somewhat|rather|quite|fairly|pretty)\s+/gi, 'DIMINISHER_');
    
    // Handle contractions
    const contractions = {
      "won't": "will not", "can't": "cannot", "n't": " not",
      "'re": " are", "'ve": " have", "'ll": " will", "'d": " would",
      "'m": " am", "it's": "it is", "that's": "that is"
    };
    
    Object.entries(contractions).forEach(([contraction, expansion]) => {
      processed = processed.replace(new RegExp(contraction, 'gi'), expansion);
    });
    
    // Remove excessive punctuation but preserve emotional indicators
    processed = processed.replace(/[!]{2,}/g, ' EXCITEMENT ');
    processed = processed.replace(/[?]{2,}/g, ' CONFUSION ');
    processed = processed.replace(/[.]{3,}/g, ' PAUSE ');
    
    // Handle emojis and emoticons
    const emojiPatterns = {
      '😊|😀|😃|😄|😁|🙂|😌|😍|🥰|😘|🤗': ' POSITIVE_EMOJI ',
      '😢|😭|😞|😔|😟|😕|🙁|☹️|😰|😨': ' NEGATIVE_EMOJI ',
      '😐|😑|🤔|😶|🙄|😏': ' NEUTRAL_EMOJI ',
      '😡|😠|🤬|😤|💢': ' ANGER_EMOJI ',
      '❤️|💕|💖|💗|💝|🧡|💛|💚|💙|💜': ' LOVE_EMOJI '
    };
    
    Object.entries(emojiPatterns).forEach(([pattern, replacement]) => {
      processed = processed.replace(new RegExp(pattern, 'g'), replacement);
    });
    
    // Handle text emoticons
    processed = processed.replace(/:\)|:-\)|:\]|:D|:-D|=\)|=D/g, ' POSITIVE_EMOTICON ');
    processed = processed.replace(/:\(|:-\(|:\[|=\(|D:/g, ' NEGATIVE_EMOTICON ');
    processed = processed.replace(/:\||:-\||=\|/g, ' NEUTRAL_EMOTICON ');
    
    return processed;
  }

  private async callSentimentAPI(text: string, modelUrl: string): Promise<ApiResponse[]> {
    try {
      const response = await axios.post(
        modelUrl,
        { inputs: text },
        {
          headers: {
            'Authorization': `Bearer ${this.apiToken}`,
            'Content-Type': 'application/json',
          },
          timeout: 15000,
        }
      );

      return response.data[0] || [];
    } catch (error) {
      console.warn(`API call failed for ${modelUrl}:`, error);
      return [];
    }
  }

  public async analyzeSentiment(text: string): Promise<ApiResponse[]> {
    const processedText = this.preprocessText(text);
    
    if (!this.apiToken || this.apiToken === 'hf_dummy_token') {
      return this.getEnhancedMockSentiment(processedText, text);
    }

    try {
      if (this.useEnsemble) {
        // Ensemble approach using multiple models
        const [primary, secondary] = await Promise.allSettled([
          this.callSentimentAPI(processedText, SENTIMENT_MODELS.primary),
          this.callSentimentAPI(processedText, SENTIMENT_MODELS.secondary)
        ]);

        const primaryResult = primary.status === 'fulfilled' ? primary.value : [];
        const secondaryResult = secondary.status === 'fulfilled' ? secondary.value : [];

        if (primaryResult.length > 0 && secondaryResult.length > 0) {
          return this.combineEnsembleResults(primaryResult, secondaryResult);
        } else if (primaryResult.length > 0) {
          return primaryResult;
        } else if (secondaryResult.length > 0) {
          return this.normalizeSecondaryModelResults(secondaryResult);
        }
      } else {
        // Single model approach
        const result = await this.callSentimentAPI(processedText, SENTIMENT_MODELS.primary);
        if (result.length > 0) {
          return result;
        }
      }
    } catch (error) {
      console.error('Ensemble analysis failed:', error);
    }

    // Fallback to enhanced mock analysis
    return this.getEnhancedMockSentiment(processedText, text);
  }

  private combineEnsembleResults(primary: ApiResponse[], secondary: ApiResponse[]): ApiResponse[] {
    // Combine results from multiple models with weighted averaging
    const labelMap: Record<string, string> = {
      'LABEL_0': 'negative',
      'LABEL_1': 'neutral',
      'LABEL_2': 'positive',
      '1 star': 'negative',
      '2 stars': 'negative',
      '3 stars': 'neutral',
      '4 stars': 'positive',
      '5 stars': 'positive'
    };

    const combinedScores: Record<string, number> = {
      'LABEL_0': 0, // negative
      'LABEL_1': 0, // neutral
      'LABEL_2': 0  // positive
    };

    // Weight primary model more heavily (70% vs 30%)
    primary.forEach(result => {
      const normalizedLabel = this.normalizeLabelToStandard(result.label);
      if (normalizedLabel) {
        combinedScores[normalizedLabel] += result.score * 0.7;
      }
    });

    secondary.forEach(result => {
      const normalizedLabel = this.normalizeLabelToStandard(result.label);
      if (normalizedLabel) {
        combinedScores[normalizedLabel] += result.score * 0.3;
      }
    });

    // Convert back to array format and sort by score
    return Object.entries(combinedScores)
      .map(([label, score]) => ({ label, score }))
      .sort((a, b) => b.score - a.score);
  }

  private normalizeLabelToStandard(label: string): string | null {
    const labelMap: Record<string, string> = {
      'LABEL_0': 'LABEL_0', // negative
      'LABEL_1': 'LABEL_1', // neutral
      'LABEL_2': 'LABEL_2', // positive
      'negative': 'LABEL_0',
      'neutral': 'LABEL_1',
      'positive': 'LABEL_2',
      '1 star': 'LABEL_0',
      '2 stars': 'LABEL_0',
      '3 stars': 'LABEL_1',
      '4 stars': 'LABEL_2',
      '5 stars': 'LABEL_2'
    };
    
    return labelMap[label] || null;
  }

  private normalizeSecondaryModelResults(results: ApiResponse[]): ApiResponse[] {
    // Convert star ratings to sentiment labels
    return results.map(result => {
      const normalizedLabel = this.normalizeLabelToStandard(result.label);
      return {
        label: normalizedLabel || result.label,
        score: result.score
      };
    }).filter(result => result.label.startsWith('LABEL_'));
  }

  private getEnhancedMockSentiment(processedText: string, originalText: string): ApiResponse[] {
    // Enhanced mock sentiment analysis with better accuracy
    const features = this.extractTextFeatures(processedText, originalText);
    
    let positiveScore = features.positiveScore;
    let negativeScore = features.negativeScore;
    let neutralScore = features.neutralScore;
    
    // Apply context-aware adjustments
    if (features.hasNegation) {
      // Flip sentiment when negation is detected
      const temp = positiveScore;
      positiveScore = negativeScore;
      negativeScore = temp;
    }
    
    if (features.hasIntensifier) {
      // Amplify the dominant sentiment
      if (positiveScore > negativeScore) {
        positiveScore *= 1.3;
      } else {
        negativeScore *= 1.3;
      }
    }
    
    if (features.hasDiminisher) {
      // Reduce sentiment intensity
      positiveScore *= 0.8;
      negativeScore *= 0.8;
      neutralScore *= 1.2;
    }
    
    // Normalize scores
    const total = positiveScore + negativeScore + neutralScore;
    if (total === 0) {
      return [
        { label: 'LABEL_1', score: 0.6 },
        { label: 'LABEL_2', score: 0.25 },
        { label: 'LABEL_0', score: 0.15 }
      ];
    }
    
    const normalizedPositive = positiveScore / total;
    const normalizedNegative = negativeScore / total;
    const normalizedNeutral = neutralScore / total;
    
    return [
      { label: 'LABEL_2', score: normalizedPositive },
      { label: 'LABEL_0', score: normalizedNegative },
      { label: 'LABEL_1', score: normalizedNeutral }
    ].sort((a, b) => b.score - a.score);
  }

  private extractTextFeatures(processedText: string, originalText: string) {
    // Enhanced feature extraction for better sentiment analysis
    const positiveWords = [
      'excellent', 'amazing', 'wonderful', 'fantastic', 'perfect', 'outstanding', 
      'brilliant', 'superb', 'magnificent', 'delightful', 'awesome', 'great',
      'good', 'love', 'like', 'enjoy', 'happy', 'pleased', 'satisfied',
      'impressive', 'remarkable', 'exceptional', 'marvelous', 'terrific'
    ];
    
    const negativeWords = [
      'terrible', 'awful', 'horrible', 'disgusting', 'disappointing', 'pathetic',
      'atrocious', 'dreadful', 'appalling', 'abysmal', 'bad', 'hate', 'dislike',
      'annoying', 'frustrating', 'useless', 'worthless', 'poor', 'worst',
      'unacceptable', 'inadequate', 'inferior', 'defective', 'faulty', 'disappointment'
      
    ];
    
    const neutralWords = [
      'okay', 'average', 'normal', 'standard', 'typical', 'regular',
      'ordinary', 'common', 'usual', 'basic', 'moderate', 'fair'
    ];
    
    const intensifiers = ['very', 'extremely', 'incredibly', 'absolutely', 'totally', 'completely'];
    const diminishers = ['slightly', 'somewhat', 'rather', 'quite', 'fairly', 'pretty'];
    
    const lowerText = originalText.toLowerCase();
    const lowerProcessed = processedText.toLowerCase();
    
    let positiveScore = 0;
    let negativeScore = 0;
    let neutralScore = 0.3; // Base neutral score
    
    // Count sentiment words with context awareness
    positiveWords.forEach(word => {
      const regex = new RegExp(`\\b${word}\\b`, 'g');
      const matches = lowerText.match(regex);
      if (matches) {
        positiveScore += matches.length * 0.4;
      }
    });
    
    negativeWords.forEach(word => {
      const regex = new RegExp(`\\b${word}\\b`, 'g');
      const matches = lowerText.match(regex);
      if (matches) {
        negativeScore += matches.length * 0.4;
      }
    });
    
    neutralWords.forEach(word => {
      const regex = new RegExp(`\\b${word}\\b`, 'g');
      const matches = lowerText.match(regex);
      if (matches) {
        neutralScore += matches.length * 0.3;
      }
    });
    
    // Check for special indicators
    const hasNegation = /NOT_/.test(lowerProcessed);
    const hasIntensifier = /INTENSIFIER_/.test(lowerProcessed);
    const hasDiminisher = /DIMINISHER_/.test(lowerProcessed);
    const hasPositiveEmoji = /POSITIVE_EMOJI|LOVE_EMOJI|POSITIVE_EMOTICON/.test(lowerProcessed);
    const hasNegativeEmoji = /NEGATIVE_EMOJI|ANGER_EMOJI|NEGATIVE_EMOTICON/.test(lowerProcessed);
    const hasExcitement = /EXCITEMENT/.test(lowerProcessed);
    
    // Apply emoji and emoticon weights
    if (hasPositiveEmoji) positiveScore += 0.5;
    if (hasNegativeEmoji) negativeScore += 0.5;
    if (hasExcitement) {
      if (positiveScore > negativeScore) positiveScore += 0.3;
      else negativeScore += 0.3;
    }
    
    // Sentence structure analysis
    const sentences = originalText.split(/[.!?]+/).filter(s => s.trim().length > 0);
    if (sentences.length > 1) {
      // Multi-sentence text - analyze each sentence
      sentences.forEach(sentence => {
        const sentenceScore = this.analyzeSentenceStructure(sentence);
        positiveScore += sentenceScore.positive;
        negativeScore += sentenceScore.negative;
        neutralScore += sentenceScore.neutral;
      });
    }
    
    return {
      positiveScore,
      negativeScore,
      neutralScore,
      hasNegation,
      hasIntensifier,
      hasDiminisher
    };
  }

  private analyzeSentenceStructure(sentence: string) {
    const words = sentence.toLowerCase().split(/\s+/);
    let positive = 0;
    let negative = 0;
    let neutral = 0;
    
    // Look for comparative structures
    if (sentence.includes('better than') || sentence.includes('worse than')) {
      if (sentence.includes('better than')) positive += 0.2;
      if (sentence.includes('worse than')) negative += 0.2;
    }
    
    // Look for conditional statements
    if (sentence.includes('if') || sentence.includes('would')) {
      neutral += 0.1; // Conditional statements are often more neutral
    }
    
    // Look for questions
    if (sentence.includes('?')) {
      neutral += 0.1; // Questions are often neutral
    }
    
    return { positive, negative, neutral };
  }

  public async analyzeBatch(texts: string[]): Promise<ApiResponse[][]> {
    const results: ApiResponse[][] = [];
    
    // Process in smaller batches to avoid rate limiting and improve accuracy
    const batchSize = 3;
    for (let i = 0; i < texts.length; i += batchSize) {
      const batch = texts.slice(i, i + batchSize);
      const batchPromises = batch.map(text => this.analyzeSentiment(text));
      const batchResults = await Promise.all(batchPromises);
      results.push(...batchResults);
      
      // Longer delay between batches for API stability
      if (i + batchSize < texts.length) {
        await new Promise(resolve => setTimeout(resolve, 2000));
      }
    }
    
    return results;
  }

  public extractKeywords(text: string, sentiment: string): string[] {
    const processedText = this.preprocessText(text);
    const words = processedText.toLowerCase().split(/\W+/).filter(word => word.length > 2);
    
    const stopWords = new Set([
      'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
      'by', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
      'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
      'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must',
      'can', 'shall', 'not_', 'intensifier_', 'diminisher_'
    ]);
    
    // Enhanced sentiment-specific keywords
    const sentimentKeywords = {
      positive: [
        'excellent', 'amazing', 'wonderful', 'fantastic', 'perfect', 'outstanding',
        'brilliant', 'superb', 'magnificent', 'delightful', 'awesome', 'great',
        'good', 'love', 'like', 'enjoy', 'happy', 'pleased', 'satisfied',
        'impressive', 'remarkable', 'exceptional', 'marvelous', 'terrific',
        'positive_emoji', 'love_emoji', 'positive_emoticon'
      ],
      negative: [
        'terrible', 'awful', 'horrible', 'disgusting', 'disappointing', 'pathetic',
        'atrocious', 'dreadful', 'appalling', 'abysmal', 'bad', 'hate', 'dislike',
        'annoying', 'frustrating', 'useless', 'worthless', 'poor', 'worst',
        'unacceptable', 'inadequate', 'inferior', 'defective', 'faulty',
        'negative_emoji', 'anger_emoji', 'negative_emoticon'
      ],
      neutral: [
        'okay', 'average', 'normal', 'standard', 'typical', 'regular',
        'ordinary', 'common', 'usual', 'basic', 'moderate', 'fair',
        'neutral_emoji', 'neutral_emoticon'
      ]
    };
    
    const relevantWords = words.filter(word => {
      if (stopWords.has(word)) return false;
      
      // Include sentiment-specific words
      if (sentimentKeywords[sentiment as keyof typeof sentimentKeywords]?.includes(word)) {
        return true;
      }
      
      // Include longer words that might be important
      if (word.length > 4) return true;
      
      // Include words that appear in context with sentiment words
      const wordIndex = words.indexOf(word);
      const context = words.slice(Math.max(0, wordIndex - 2), wordIndex + 3);
      const hasSentimentContext = context.some(contextWord => 
        sentimentKeywords.positive.includes(contextWord) ||
        sentimentKeywords.negative.includes(contextWord) ||
        sentimentKeywords.neutral.includes(contextWord)
      );
      
      return hasSentimentContext;
    });
    
    // Remove duplicates and limit to most relevant keywords
    const uniqueKeywords = [...new Set(relevantWords)];
    
    // Score keywords by relevance and frequency
    const keywordScores = uniqueKeywords.map(keyword => {
      let score = 0;
      
      // Higher score for sentiment-specific words
      if (sentimentKeywords[sentiment as keyof typeof sentimentKeywords]?.includes(keyword)) {
        score += 3;
      }
      
      // Score by frequency
      const frequency = words.filter(w => w === keyword).length;
      score += frequency;
      
      // Score by length (longer words often more meaningful)
      score += Math.min(keyword.length / 10, 1);
      
      return { keyword, score };
    });
    
    // Return top keywords sorted by score
    return keywordScores
      .sort((a, b) => b.score - a.score)
      .slice(0, 6)
      .map(item => item.keyword)
      .filter(keyword => !keyword.includes('_')); // Remove processed markers
  }
}

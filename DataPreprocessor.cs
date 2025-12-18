using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace CryptoPredictor
{
    public class DataPreprocessor
    {
        public static List<CryptoTimeSeriesData> EnrichTimeSeriesData(List<CryptoTimeSeriesData> rawData)
        {
            if (rawData == null || !rawData.Any())
                return new List<CryptoTimeSeriesData>();

            // Sort by timestamp
            var sortedData = rawData.OrderBy(d => d.Timestamp).ToList();
            
            // Group by symbol
            var symbolGroups = sortedData.GroupBy(d => d.Symbol);
            
            var enrichedData = new List<CryptoTimeSeriesData>();
            
            foreach (var group in symbolGroups)
            {
                var symbolData = group.ToList();
                
                // Note: BTC ratio precomputation removed; PriceToBtc is disabled for now.

                // We'll compute volatility values, then compute z-scores for the symbol once all values exist
                var tempList = new List<CryptoTimeSeriesData>();
                for (int i = 0; i < symbolData.Count; i++)
                {
                    var currentRecord = symbolData[i];

                    // Calculate price change percentage first
                    if (i > 0)
                    {
                        float previousClose = symbolData[i-1].ClosePrice;
                        float currentClose = currentRecord.ClosePrice;
                        currentRecord.PriceChangePercentage = (currentClose - previousClose) / previousClose * 100;
                        currentRecord.PreviousDayChange = symbolData[i-1].PriceChangePercentage;
                    }
                    else
                    {
                        currentRecord.PriceChangePercentage = 0;
                        currentRecord.PreviousDayChange = 0;
                    }

                    // Calculate moving averages (null until enough history)
                    currentRecord.MovingAverage5Day = CalculateMovingAverage(symbolData, i, 5);
                    currentRecord.MovingAverage20Day = CalculateMovingAverage(symbolData, i, 20);

                    // Calculate RSI (null until enough history)
                    currentRecord.RelativeStrengthIndex = CalculateRSI(symbolData, i, 14);

                    // Calculate volatility at multiple scales (sample standard deviation of returns in percent)
                    currentRecord.Volatility7 = CalculateRollingStd(symbolData, i, 7);
                    currentRecord.Volatility14 = CalculateRollingStd(symbolData, i, 14);
                    currentRecord.Volatility21 = CalculateRollingStd(symbolData, i, 21);

                    // PriceToBtc is withheld for now (disabled)

                    tempList.Add(currentRecord);
                }

                // Compute z-scores (standardize) per symbol using historical mean/std (sample)
                var v7vals = tempList.Select(t => t.Volatility7).Where(v => v.HasValue).Select(v => v.Value).ToList();
                var v14vals = tempList.Select(t => t.Volatility14).Where(v => v.HasValue).Select(v => v.Value).ToList();
                var v21vals = tempList.Select(t => t.Volatility21).Where(v => v.HasValue).Select(v => v.Value).ToList();

                float mean7 = v7vals.Any() ? (float)v7vals.Average() : 0f; float std7 = v7vals.Count > 1 ? (float)Math.Sqrt(v7vals.Sum(v => Math.Pow(v - mean7, 2)) / (v7vals.Count - 1)) : 0f;
                float mean14 = v14vals.Any() ? (float)v14vals.Average() : 0f; float std14 = v14vals.Count > 1 ? (float)Math.Sqrt(v14vals.Sum(v => Math.Pow(v - mean14, 2)) / (v14vals.Count - 1)) : 0f;
                float mean21 = v21vals.Any() ? (float)v21vals.Average() : 0f; float std21 = v21vals.Count > 1 ? (float)Math.Sqrt(v21vals.Sum(v => Math.Pow(v - mean21, 2)) / (v21vals.Count - 1)) : 0f;

                foreach (var rec in tempList)
                {
                    rec.Volatility7Z = std7 == 0 ? 0f : ((rec.Volatility7 ?? 0f) - mean7) / std7;
                    rec.Volatility14Z = std14 == 0 ? 0f : ((rec.Volatility14 ?? 0f) - mean14) / std14;
                    rec.Volatility21Z = std21 == 0 ? 0f : ((rec.Volatility21 ?? 0f) - mean21) / std21;

                    enrichedData.Add(rec);
                }
            }
            
            return enrichedData;
        }
        
    private static float? CalculateMovingAverage(List<CryptoTimeSeriesData> data, int currentIndex, int window)
        {
            if (currentIndex < window - 1)
        return null;
                
            float sum = 0;
            for (int i = 0; i < window; i++)
            {
                sum += data[currentIndex - i].ClosePrice;
            }
            
            return sum / window;
        }
        
    private static float? CalculateRSI(List<CryptoTimeSeriesData> data, int currentIndex, int window)
        {
            // Simple RSI implementation
            if (currentIndex < window)
        return null; // Not enough history for RSI
                
            float gainSum = 0;
            float lossSum = 0;
            
            for (int i = currentIndex - window + 1; i <= currentIndex; i++)
            {
                float change = i > 0 ? data[i].ClosePrice - data[i-1].ClosePrice : 0;
                if (change >= 0)
                    gainSum += change;
                else
                    lossSum -= change;
            }
            
            float avgGain = gainSum / window;
            float avgLoss = lossSum / window;
            
            if (avgLoss == 0)
                return 100;
                
            float rs = avgGain / avgLoss;
            return 100 - (100 / (1 + rs));
        }

        private static float? CalculateRollingStd(List<CryptoTimeSeriesData> data, int currentIndex, int window)
        {
            if (currentIndex < window - 1)
                return null; // Not enough history

            var values = new List<float>();
            for (int i = currentIndex - window + 1; i <= currentIndex; i++)
            {
                // Use the already-computed PriceChangePercentage (in percent) when available
                if (data[i].PriceChangePercentage.HasValue)
                    values.Add(data[i].PriceChangePercentage.Value);
            }

            if (values.Count <= 1) return 0;

            var mean = values.Average();
            double sumSq = values.Sum(v => Math.Pow(v - mean, 2));
            var variance = sumSq / (values.Count - 1); // sample variance
            return (float)Math.Sqrt(variance);
        }
    }
}
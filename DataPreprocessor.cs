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
                    
                    enrichedData.Add(currentRecord);
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
    }
}
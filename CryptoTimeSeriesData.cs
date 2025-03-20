using System;

namespace CryptoPredictor
{
    public class CryptoTimeSeriesData
    {
        public string Symbol { get; set; } = string.Empty;
        public DateTime Timestamp { get; set; }
        public float OpenPrice { get; set; }
        public float HighPrice { get; set; }
        public float LowPrice { get; set; }
        public float ClosePrice { get; set; }
        public float Volume { get; set; }
        
        // Feature engineering
        public float PriceChange => ClosePrice - OpenPrice;
        public float PriceChangePercentage => (ClosePrice - OpenPrice) / OpenPrice * 100;
        public float VolatilityIndicator => HighPrice - LowPrice;

        // Window features (to be populated by preprocessing)
        public float PreviousDayChange { get; set; }
        public float MovingAverage5Day { get; set; }
        public float MovingAverage20Day { get; set; }
        public float RelativeStrengthIndex { get; set; }
    }

    public class CryptoTimeSeriesPrediction
    {
        public float PredictedClosePrice { get; set; }
        public float PredictedHighPrice { get; set; }
        public float PredictedLowPrice { get; set; }
    }
}
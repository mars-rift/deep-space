using System;

namespace CryptoPredictor
{
    public class CryptoTimeSeriesData
    {
        // Basic price data
        public string Symbol { get; set; } = string.Empty;
        public DateTime Timestamp { get; set; }
        public float OpenPrice { get; set; }
        public float HighPrice { get; set; }
        public float LowPrice { get; set; }
        public float ClosePrice { get; set; }
        public float Volume { get; set; }

        // Properties expected by DataPreprocessor
        public float? PreviousDayChange { get; set; }
        public float? PriceChangePercentage { get; set; }
        public float? MovingAverage5Day { get; set; }
        public float? MovingAverage20Day { get; set; }
        public float? RelativeStrengthIndex { get; set; }

        // Optional properties for compatibility with other code
        public float? MA7 { get; set; }
        public float? MA14 { get; set; }
        public float? MA30 { get; set; }
        public float? RSI { get; set; }
    }
}
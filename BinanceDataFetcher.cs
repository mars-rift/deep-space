using System;
using System.Collections.Generic;
using System.Net.Http;
using System.Net.Http.Json;
using System.Threading.Tasks;
using System.Linq;
using System.Text.Json;
using System.Globalization;

namespace CryptoPredictor
{
    public class BinanceDataFetcher
    {
        private readonly HttpClient _httpClient;
        private const string BaseUrl = "https://api.binance.com/api/v3";

        public BinanceDataFetcher()
        {
            _httpClient = new HttpClient();
            _httpClient.DefaultRequestHeaders.Add("User-Agent", "CryptoPredictorApp");
        }

        public async Task<List<CryptoTimeSeriesData>> GetBitcoinOHLCVAsync(string interval = "1d", int limit = 365)
        {
            try
            {
                Console.WriteLine($"Fetching Bitcoin OHLCV data from Binance (interval: {interval}, limit: {limit})...");
                var url = $"{BaseUrl}/klines?symbol=BTCUSDT&interval={interval}&limit={limit}";
                var response = await _httpClient.GetStringAsync(url);
                var candles = JsonSerializer.Deserialize<List<JsonElement[]>>(response);
                
                if (candles == null)
                    return new List<CryptoTimeSeriesData>();

                var result = new List<CryptoTimeSeriesData>();
                foreach (var candle in candles)
                {
                    if (candle.Length < 6) continue;

                    // Convert timestamp from milliseconds to DateTime
                    var timestamp = DateTimeOffset.FromUnixTimeMilliseconds(candle[0].GetInt64()).DateTime;
                    
                    result.Add(new CryptoTimeSeriesData
                    {
                        Symbol = "BTC",
                        Timestamp = timestamp,
                        OpenPrice = Convert.ToSingle(candle[1].GetString(), CultureInfo.InvariantCulture),
                        HighPrice = Convert.ToSingle(candle[2].GetString(), CultureInfo.InvariantCulture),
                        LowPrice = Convert.ToSingle(candle[3].GetString(), CultureInfo.InvariantCulture),
                        ClosePrice = Convert.ToSingle(candle[4].GetString(), CultureInfo.InvariantCulture),
                        Volume = Convert.ToSingle(candle[5].GetString(), CultureInfo.InvariantCulture)
                    });
                }

                // Return in chronological order
                return result.OrderBy(d => d.Timestamp).ToList();
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error fetching Bitcoin OHLCV data: {ex.Message}");
                if (ex.InnerException != null)
                {
                    Console.WriteLine($"Inner exception: {ex.InnerException.Message}");
                }
                return new List<CryptoTimeSeriesData>();
            }
        }

        // Simple method to get price data for regular model (non-time series)
        public async Task<List<CryptoData>> GetBitcoinPricesAsync(int limit = 365)
        {
            var ohlcvData = await GetBitcoinOHLCVAsync("1d", limit);
            
            // Convert OHLCV data to simple price data
            return ohlcvData.Select(d => new CryptoData { 
                Symbol = d.Symbol, 
                Price = d.ClosePrice 
            }).ToList();
        }
    }
}
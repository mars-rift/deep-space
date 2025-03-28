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
    public class CoinGeckoDataFetcher
    {
        private readonly HttpClient _httpClient;
        private const string BaseUrl = "https://api.coingecko.com/api/v3";

        public CoinGeckoDataFetcher()
        {
            _httpClient = new HttpClient();
            _httpClient.DefaultRequestHeaders.Add("User-Agent", "CryptoPredictorApp");
        }

        public async Task<List<CryptoTimeSeriesData>> GetBitcoinOHLCVAsync(int days = 365)
        {
            try
            {
                Console.WriteLine($"Fetching Bitcoin OHLCV data from CoinGecko (days: {days})...");
                
                // CoinGecko API for Bitcoin market data
                var url = $"{BaseUrl}/coins/bitcoin/market_chart?vs_currency=usd&days={days}&interval=daily";
                var response = await _httpClient.GetStringAsync(url);
                
                // Parse the response
                using var jsonDoc = JsonDocument.Parse(response);
                var root = jsonDoc.RootElement;
                
                // Extract the prices, which are [timestamp, price] pairs
                var pricesArray = root.GetProperty("prices");
                var volumesArray = root.GetProperty("total_volumes");
                
                var result = new List<CryptoTimeSeriesData>();
                
                // Convert the data to our format
                for (int i = 0; i < pricesArray.GetArrayLength(); i++)
                {
                    // Skip if we don't have complete data
                    if (i >= volumesArray.GetArrayLength()) continue;
                    
                    // Get timestamp and price
                    var priceData = pricesArray[i];
                    var timestamp = DateTimeOffset.FromUnixTimeMilliseconds((long)priceData[0].GetDouble()).DateTime;
                    var closePrice = (float)priceData[1].GetDouble();
                    
                    // Get volume
                    var volumeData = volumesArray[i];
                    var volume = (float)volumeData[1].GetDouble();
                    
                    // CoinGecko only provides daily close prices, not OHLC
                    // For simplicity, we'll use the close price for all fields
                    result.Add(new CryptoTimeSeriesData
                    {
                        Symbol = "BTC",
                        Timestamp = timestamp,
                        OpenPrice = closePrice,    // Using close as open since we only have daily data
                        HighPrice = closePrice,    // Using close as high since we only have daily data
                        LowPrice = closePrice,     // Using close as low since we only have daily data
                        ClosePrice = closePrice,
                        Volume = volume
                    });
                }

                // Return in chronological order
                return result.OrderBy(d => d.Timestamp).ToList();
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error fetching Bitcoin data from CoinGecko: {ex.Message}");
                if (ex.InnerException != null)
                {
                    Console.WriteLine($"Inner exception: {ex.InnerException.Message}");
                }
                return new List<CryptoTimeSeriesData>();
            }
        }

        // Simple method to get price data for regular model (non-time series)
        public async Task<List<CryptoData>> GetBitcoinPricesAsync(int days = 365)
        {
            var ohlcvData = await GetBitcoinOHLCVAsync(days);
            
            // Convert OHLCV data to simple price data
            return ohlcvData.Select(d => new CryptoData { 
                Symbol = d.Symbol, 
                Price = d.ClosePrice 
            }).ToList();
        }
    }
}
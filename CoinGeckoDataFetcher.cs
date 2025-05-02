using System;
using System.Collections.Generic;
using System.Net.Http;
using System.Text.Json;
using System.Threading.Tasks;
using System.Linq;

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

        // Fetch OHLCV time series data for any coin
        public async Task<List<CryptoTimeSeriesData>> GetOHLCVAsync(string coinGeckoId, int days = 365)
        {
            try
            {
                Console.WriteLine($"Fetching OHLCV data from CoinGecko for {coinGeckoId} ({days} days)...");
                var url = $"{BaseUrl}/coins/{coinGeckoId}/market_chart?vs_currency=usd&days={days}&interval=daily";
                var response = await _httpClient.GetStringAsync(url);

                using var jsonDoc = JsonDocument.Parse(response);
                var root = jsonDoc.RootElement;

                var pricesArray = root.GetProperty("prices");
                var volumesArray = root.GetProperty("total_volumes");

                var result = new List<CryptoTimeSeriesData>();

                for (int i = 0; i < pricesArray.GetArrayLength(); i++)
                {
                    if (i >= volumesArray.GetArrayLength()) continue;

                    var priceData = pricesArray[i];
                    var timestamp = DateTimeOffset.FromUnixTimeMilliseconds((long)priceData[0].GetDouble()).DateTime;
                    var closePrice = (float)priceData[1].GetDouble();

                    var volumeData = volumesArray[i];
                    var volume = (float)volumeData[1].GetDouble();

                    result.Add(new CryptoTimeSeriesData
                    {
                        Symbol = coinGeckoId.ToUpper(),
                        Timestamp = timestamp,
                        OpenPrice = closePrice,
                        HighPrice = closePrice,
                        LowPrice = closePrice,
                        ClosePrice = closePrice,
                        Volume = volume
                    });
                }

                return result.OrderBy(d => d.Timestamp).ToList();
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error fetching data from CoinGecko: {ex.Message}");
                if (ex.InnerException != null)
                {
                    Console.WriteLine($"Inner exception: {ex.InnerException.Message}");
                }
                return new List<CryptoTimeSeriesData>();
            }
        }

        // Fetch simple price data for any coin
        public async Task<List<CryptoData>> GetPricesAsync(string coinGeckoId, int days = 365)
        {
            var ohlcvData = await GetOHLCVAsync(coinGeckoId, days);
            return ohlcvData.Select(d => new CryptoData
            {
                Symbol = d.Symbol,
                Price = d.ClosePrice
            }).ToList();
        }
    }
}
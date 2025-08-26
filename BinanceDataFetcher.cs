using System;
using System.Collections.Generic;
using System.Net.Http;
using System.Threading.Tasks;
using System.Linq;
using System.Text.Json;
using System.Globalization;

namespace CryptoPredictor
{
    public class BinanceDataFetcher
    {
        private readonly HttpClient _httpClient;
        private const string BaseUrl = "https://api.binance.us/api/v3";

        public BinanceDataFetcher()
        {
            _httpClient = new HttpClient();
            _httpClient.DefaultRequestHeaders.Add("User-Agent", "CryptoPredictorApp");
        }

        // Fetch OHLCV time series data for any symbol (e.g., "ETHUSDT")
        public async Task<List<CryptoTimeSeriesData>> GetOHLCVAsync(string binanceSymbol, string interval = "1d", int limit = 365)
        {
            try
            {
                Console.WriteLine($"Fetching OHLCV data from Binance for {binanceSymbol} (interval: {interval}, limit: {limit})...");
                var url = $"{BaseUrl}/klines?symbol={binanceSymbol}&interval={interval}&limit={limit}";
                var response = await _httpClient.GetStringAsync(url);
                var candles = JsonSerializer.Deserialize<List<JsonElement[]>>(response);

                if (candles == null)
                    return new List<CryptoTimeSeriesData>();

                var result = new List<CryptoTimeSeriesData>();
                foreach (var candle in candles)
                {
                    if (candle.Length < 6) continue;

                    var timestamp = DateTimeOffset.FromUnixTimeMilliseconds(candle[0].GetInt64()).UtcDateTime;

                    result.Add(new CryptoTimeSeriesData
                    {
                        // Derive a reasonable symbol label from the trading pair (e.g., ETHUSDT -> ETH)
                        Symbol = DeriveBaseSymbol(binanceSymbol),
                        Timestamp = timestamp,
                        OpenPrice = Convert.ToSingle(candle[1].GetString(), CultureInfo.InvariantCulture),
                        HighPrice = Convert.ToSingle(candle[2].GetString(), CultureInfo.InvariantCulture),
                        LowPrice = Convert.ToSingle(candle[3].GetString(), CultureInfo.InvariantCulture),
                        ClosePrice = Convert.ToSingle(candle[4].GetString(), CultureInfo.InvariantCulture),
                        Volume = Convert.ToSingle(candle[5].GetString(), CultureInfo.InvariantCulture)
                    });
                }

                return result.OrderBy(d => d.Timestamp).ToList();
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error fetching OHLCV data for {binanceSymbol}: {ex.Message}");
                if (ex.InnerException != null)
                {
                    Console.WriteLine($"Inner exception: {ex.InnerException.Message}");
                }
                return new List<CryptoTimeSeriesData>();
            }
        }

        // Fetch simple price data for any symbol (e.g., "ETHUSDT")
        public async Task<List<CryptoData>> GetPricesAsync(string binanceSymbol, int limit = 365)
        {
            var ohlcvData = await GetOHLCVAsync(binanceSymbol, "1d", limit);
            return ohlcvData.Select(d => new CryptoData
            {
                Symbol = d.Symbol,
                Price = d.ClosePrice
            }).ToList();
        }

        private static string DeriveBaseSymbol(string tradingPair)
        {
            if (string.IsNullOrWhiteSpace(tradingPair)) return "";
            // Common quote assets by length (longest first to ensure proper matching)
            var quotes = new[] { "BUSD", "USDT", "USDC", "FDUSD", "TUSD", "BTC", "ETH", "BNB", "EUR", "TRY", "BRL", "GBP", "AUD", "RUB", "JPY" };
            foreach (var q in quotes)
            {
                if (tradingPair.EndsWith(q, StringComparison.OrdinalIgnoreCase))
                {
                    return tradingPair[..^q.Length].ToUpperInvariant();
                }
            }
            // Fallback: return input uppercased
            return tradingPair.ToUpperInvariant();
        }
    }
}
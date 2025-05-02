using Microsoft.ML;
using Microsoft.ML.Data;
using CsvHelper;
using CsvHelper.Configuration;
using System.Globalization;
using System.Linq;
using System.Collections.Generic;
using System.IO;
using System;
using System.Threading.Tasks;

namespace CryptoPredictor
{
    public class Program
    {
        private const string CSV_FILE_PATH = "crypto.csv";
        private const string TIMESERIES_CSV_PATH = "crypto_timeseries.csv"; // For time series data
        private const string OUTPUT_DIR = "output";
        private const string DEFAULT_SYMBOL = "ETH";
        private const string DEFAULT_COINGECKO_ID = "ethereum";
        private const string DEFAULT_BINANCE_SYMBOL = "ETHUSDT";

        public static async Task Main(string[] args)
        {
            string symbol = DEFAULT_SYMBOL;
            string coinGeckoId = DEFAULT_COINGECKO_ID;
            string binanceSymbol = DEFAULT_BINANCE_SYMBOL;

            try
            {
                // Create output directory if it doesn't exist
                if (!Directory.Exists(OUTPUT_DIR))
                    Directory.CreateDirectory(OUTPUT_DIR);

                Console.WriteLine("=== Crypto Price Predictor ===");

                // Load and prepare regular data from CoinGecko
                Console.WriteLine($"Loading basic price data for {symbol} from CoinGecko...");
                var cryptoData = await LoadDataAsync(coinGeckoId, binanceSymbol);
                if (cryptoData?.Any() != true)
                {
                    Console.WriteLine("Error: No data found from Binance API.");
                    return;
                }
                Console.WriteLine($"Loaded {cryptoData.Count} price records from Binance");

                // Load time series data
                Console.WriteLine($"Loading time series data for {symbol}...");
                var timeSeriesData = await LoadTimeSeriesDataAsync(coinGeckoId, binanceSymbol);
                if (timeSeriesData?.Any() == true)
                {
                    Console.WriteLine($"Loaded {timeSeriesData.Count} time series records from Binance");

                    // Rest of your code stays the same
                    Console.WriteLine("Enriching time series data with technical indicators...");
                    var enrichedData = DataPreprocessor.EnrichTimeSeriesData(timeSeriesData);

                    // Train the model using enriched data
                    var model = TrainModel(enrichedData);

                    Console.WriteLine($"Enrichment complete: {enrichedData.Count} records processed");

                    // Create visualizations for the first symbol in the dataset
                    if (enrichedData.Any())
                    {
                        var firstSymbol = enrichedData.First().Symbol;
                        Console.WriteLine($"Creating time series visualization for {firstSymbol}...");
                        CreateTimeSeriesVisualization(enrichedData, firstSymbol);
                    }

                    // Generate residuals for visualization
                    var residuals = GenerateResiduals(model, enrichedData);

                    // Add timestamp to filename to avoid overwriting
                    string timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
                    string predictionVsActualPath = Path.Combine(OUTPUT_DIR, $"prediction_vs_actual_{timestamp}.png");

                    // Create visualization for prediction vs actual
                    if (residuals.Any())
                    {
                        Console.WriteLine("Creating prediction vs actual visualization...");
                        Visualization.CreatePredictionVsActualChart(
                            residuals,
                            predictionVsActualPath
                        );

                        Console.WriteLine($"Prediction vs actual chart saved to: {Path.GetFullPath(predictionVsActualPath)}");
                    }

                    // Use the model for predictions (example)
                    var latestIndicators = enrichedData.Last(); // latest enriched data point
                    var maxPrice = enrichedData.Max(x => x.ClosePrice);

                    var prediction = Predict(model, new EnhancedCryptoData
                    {
                        Symbol = symbol,
                        Price = latestIndicators.ClosePrice,
                        LogPrice = (float)Math.Log(Math.Max(1, latestIndicators.ClosePrice)),
                        PriceRatio = latestIndicators.ClosePrice / maxPrice,
                        PriceSquared = latestIndicators.ClosePrice * latestIndicators.ClosePrice,
                        MovingAverage5Day = latestIndicators.MovingAverage5Day.Value,
                        MovingAverage20Day = latestIndicators.MovingAverage20Day.Value,
                        RelativeStrengthIndex = latestIndicators.RelativeStrengthIndex.Value
                    });

                    Console.WriteLine($"Predicted Best Price to Buy ETH: {prediction.PredictedPrice:C2}");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"An unexpected error occurred: {ex.Message}");
                if (ex.InnerException != null)
                {
                    Console.WriteLine($"Inner exception: {ex.InnerException.Message}");
                }
            }
        }

        private static async Task<List<CryptoData>> LoadDataAsync(string coinGeckoId, string binanceSymbol)
        {
            try
            {
                // Try CoinGecko first
                var coinGeckoDataFetcher = new CoinGeckoDataFetcher();
                var coinGeckoData = await coinGeckoDataFetcher.GetPricesAsync(coinGeckoId, 365);
                if (coinGeckoData?.Any() == true)
                {
                    Console.WriteLine("Successfully loaded data from CoinGecko!");
                    return CleanCryptoData(coinGeckoData);
                }
                Console.WriteLine("No data returned from CoinGecko, trying Binance...");

                // Then try Binance as backup
                var binanceDataFetcher = new BinanceDataFetcher();
                var data = await binanceDataFetcher.GetPricesAsync(binanceSymbol, 365);
                if (data?.Any() == true)
                {
                    Console.WriteLine("Successfully loaded data from Binance!");
                    return CleanCryptoData(data);
                }

                // If both fail, fall back to CSV
                Console.WriteLine("Both online APIs failed, falling back to CSV data...");
                return CleanCryptoData(LoadData());
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error loading online data: {ex.Message}");
                Console.WriteLine("Falling back to CSV data...");
                return CleanCryptoData(LoadData());
            }
        }

        private static List<CryptoData> LoadData()
        {
            var config = new CsvConfiguration(CultureInfo.InvariantCulture)
            {
                HasHeaderRecord = true,
                TrimOptions = TrimOptions.Trim,
                MissingFieldFound = null,
                PrepareHeaderForMatch = args => args.Header.ToLower(),
                DetectDelimiter = true
            };

            if (!File.Exists(CSV_FILE_PATH))
            {
                throw new FileNotFoundException($"CSV file not found at path: {CSV_FILE_PATH}");
            }

            try
            {
                using var reader = new StreamReader(CSV_FILE_PATH);
                using var csv = new CsvReader(reader, config);
                csv.Context.RegisterClassMap<CryptoDataMap>();
                return csv.GetRecords<CryptoData>().ToList();
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error loading data: {ex.Message}");
                throw; // Rethrow to be handled by the main error handler
            }
        }

        private static async Task<List<CryptoTimeSeriesData>> LoadTimeSeriesDataAsync(string coinGeckoId, string binanceSymbol)
        {
            try
            {
                // First try Binance
                var binanceDataFetcher = new BinanceDataFetcher();
                var data = await binanceDataFetcher.GetOHLCVAsync(binanceSymbol, "1d", 365);
                if (data?.Any() == true)
                {
                    Console.WriteLine("Successfully loaded time series data from Binance!");
                    return data;
                }
                Console.WriteLine("No time series data returned from Binance, trying CoinGecko...");

                // Then try CoinGecko as backup
                var coinGeckoDataFetcher = new CoinGeckoDataFetcher();
                var coinGeckoData = await coinGeckoDataFetcher.GetOHLCVAsync(coinGeckoId, 365);
                if (coinGeckoData?.Any() == true)
                {
                    Console.WriteLine("Successfully loaded time series data from CoinGecko!");
                    return coinGeckoData;
                }

                // If both fail, fall back to CSV
                Console.WriteLine("Both online APIs failed, falling back to CSV time series data...");
                return LoadTimeSeriesData();
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error loading online time series data: {ex.Message}");
                Console.WriteLine("Falling back to CSV time series data...");
                return LoadTimeSeriesData();
            }
        }

        private static List<CryptoTimeSeriesData> LoadTimeSeriesData()
        {
            if (!File.Exists(TIMESERIES_CSV_PATH))
            {
                Console.WriteLine($"Time series data file not found at path: {TIMESERIES_CSV_PATH}");
                return new List<CryptoTimeSeriesData>();
            }

            try
            {
                var config = new CsvConfiguration(CultureInfo.InvariantCulture)
                {
                    HasHeaderRecord = true,
                    TrimOptions = TrimOptions.Trim,
                    MissingFieldFound = null,
                    PrepareHeaderForMatch = args => args.Header.ToLower(),
                    DetectDelimiter = true
                };

                using var reader = new StreamReader(TIMESERIES_CSV_PATH);
                using var csv = new CsvReader(reader, config);

                // Register class mapping for time series data
                csv.Context.RegisterClassMap<CryptoTimeSeriesDataMap>();

                return csv.GetRecords<CryptoTimeSeriesData>().ToList();
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error loading time series data: {ex.Message}");
                return new List<CryptoTimeSeriesData>();
            }
        }

        private static ITransformer TrainModel(List<CryptoTimeSeriesData> enrichedData)
        {
            var context = new MLContext(seed: 42);

            try
            {
                // Validate data
                var trainingData = enrichedData
                    .Where(d => d.MovingAverage5Day.HasValue && d.MovingAverage20Day.HasValue && d.RelativeStrengthIndex.HasValue)
                    .Select(d => new EnhancedCryptoData
                    {
                        Symbol = d.Symbol,
                        Price = d.ClosePrice,
                        LogPrice = (float)Math.Log(Math.Max(1, d.ClosePrice)),
                        PriceRatio = d.ClosePrice / enrichedData.Max(x => x.ClosePrice),
                        PriceSquared = d.ClosePrice * d.ClosePrice,
                        MovingAverage5Day = d.MovingAverage5Day.Value,
                        MovingAverage20Day = d.MovingAverage20Day.Value,
                        RelativeStrengthIndex = d.RelativeStrengthIndex.Value
                    }).ToList();

                var dataView = context.Data.LoadFromEnumerable(trainingData);

                // Pipeline with technical indicators
                var pipeline = context.Transforms.ReplaceMissingValues(new[] {
                        new InputOutputColumnPair("Price"),
                        new InputOutputColumnPair("LogPrice"),
                        new InputOutputColumnPair("PriceRatio"),
                        new InputOutputColumnPair("PriceSquared"),
                        new InputOutputColumnPair("MovingAverage5Day"),
                        new InputOutputColumnPair("MovingAverage20Day"),
                        new InputOutputColumnPair("RelativeStrengthIndex")
                    })
                    .Append(context.Transforms.Categorical.OneHotEncoding("SymbolEncoded", "Symbol"))
                    .Append(context.Transforms.Concatenate("Features",
                        "LogPrice", "PriceRatio", "PriceSquared",
                        "MovingAverage5Day", "MovingAverage20Day", "RelativeStrengthIndex",
                        "SymbolEncoded"))
                    .Append(context.Transforms.NormalizeMinMax("Features"))
                    .Append(context.Regression.Trainers.Sdca(
                        labelColumnName: "Price",
                        featureColumnName: "Features",
                        maximumNumberOfIterations: 100));

                var splitData = context.Data.TrainTestSplit(dataView, testFraction: 0.2, seed: 42);

                Console.WriteLine("Starting model training...");
                var model = pipeline.Fit(splitData.TrainSet);
                Console.WriteLine("Model training completed successfully");

                return model;
            }
            catch (Exception ex)
            {
                throw new InvalidOperationException($"Model training failed: {ex.Message}", ex);
            }
        }

        private static void PrintFeatureImportance(MLContext context, ITransformer model, IDataView data)
        {
            try
            {
                // Transform the input data to ensure Features column exists
                var transformedData = model.Transform(data);

                // Get the schema to verify column names
                var schema = transformedData.Schema;
                if (schema.GetColumnOrNull("Features") == null)
                {
                    Console.WriteLine("\nAvailable columns in schema:");
                    foreach (var column in schema)
                    {
                        Console.WriteLine($"- {column.Name} ({column.Type})");
                    }
                    throw new InvalidOperationException("Features column not found in the transformed data schema");
                }

                var permutationMetrics = context.Regression.PermutationFeatureImportance(
                    model,
                    transformedData,  // Use transformed data instead of original
                    labelColumnName: "Price");

                Console.WriteLine("\nFeature Importance:");
                foreach (var metric in permutationMetrics)
                {
                    if (metric.Value?.RootMeanSquaredError?.Mean != null)
                    {
                        Console.WriteLine($"Feature {metric.Key}: {metric.Value.RootMeanSquaredError.Mean:F4}");
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Feature importance calculation error: {ex.Message}");
                // Add schema information to help diagnose the issue
                try
                {
                    Console.WriteLine("\nAvailable columns in schema:");
                    foreach (var column in data.Schema)
                    {
                        Console.WriteLine($"- {column.Name} ({column.Type})");
                    }
                }
                catch
                {
                    // Ignore schema printing errors
                }
                if (ex.InnerException != null)
                {
                    Console.WriteLine($"Inner exception: {ex.InnerException.Message}");
                }
            }
        }

        private static void PlotResiduals(IDataView predictions)
        {
            try
            {
                // Create a class to match the schema
                var mlContext = new MLContext();
                var schema = predictions.Schema;

                var predictionCol = schema.GetColumnOrNull("Score")?.Index ?? -1;
                var labelCol = schema.GetColumnOrNull("Price")?.Index ?? -1;

                if (predictionCol == -1 || labelCol == -1)
                {
                    throw new InvalidOperationException("Required columns not found in predictions");
                }

                // Use cursor-based approach for better performance
                var cursor = predictions.GetRowCursor(predictions.Schema);
                var residualsList = new List<(float actual, float predicted, float residual)>();

                var labelGetter = cursor.GetGetter<float>(predictions.Schema[labelCol]);
                var predictionGetter = cursor.GetGetter<float>(predictions.Schema[predictionCol]);

                while (cursor.MoveNext())
                {
                    float label = 0;
                    float prediction = 0;
                    labelGetter(ref label);
                    predictionGetter(ref prediction);

                    if (!float.IsNaN(label) && !float.IsNaN(prediction))
                    {
                        residualsList.Add((label, prediction, label - prediction));
                    }

                    if (residualsList.Count >= 100) break; // Limit to first 100 records
                }

                if (residualsList.Any())
                {
                    Console.WriteLine("\nResiduals Analysis:");
                    foreach (var (actual, predicted, residual) in residualsList.Take(10))
                    {
                        Console.WriteLine(
                            $"Actual: {actual:F2}, " +
                            $"Predicted: {predicted:F2}, " +
                            $"Residual: {residual:F2}");
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Residuals analysis error: {ex.Message}");
                if (ex.InnerException != null)
                {
                    Console.WriteLine($"Inner exception: {ex.InnerException.Message}");
                }
            }
        }

        private static CryptoPrediction Predict(ITransformer model, EnhancedCryptoData input)
        {
            var context = new MLContext();
            var predictionEngine = context.Model.CreatePredictionEngine<EnhancedCryptoData, CryptoPrediction>(model);
            return predictionEngine.Predict(input);
        }

        private static void CreateTimeSeriesVisualization(List<CryptoTimeSeriesData> data, string symbol)
        {
            try
            {
                var filteredData = data.Where(d => d.Symbol == symbol).OrderBy(d => d.Timestamp).ToList();

                if (!filteredData.Any())
                {
                    Console.WriteLine($"No time series data found for {symbol}");
                    return;
                }

                // Add timestamp to filename to avoid overwriting
                string timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
                string outputPath = Path.Combine(OUTPUT_DIR, $"{symbol}_price_chart_{timestamp}.png");

                Visualization.CreatePriceChart(filteredData.Select(d => new CryptoData
                {
                    Symbol = d.Symbol,
                    Price = d.ClosePrice
                }).ToList(), symbol, outputPath);

                Console.WriteLine($"Time series chart saved to: {Path.GetFullPath(outputPath)}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error creating time series visualization: {ex.Message}");
            }
        }

        private static List<ResidualData> GenerateResiduals(ITransformer model, List<CryptoTimeSeriesData> enrichedData)
        {
            var context = new MLContext();
            var residuals = new List<ResidualData>();

            try
            {
                // Ensure we have valid technical indicators
                var validData = enrichedData
                    .Where(d => d.MovingAverage5Day.HasValue && d.MovingAverage20Day.HasValue && d.RelativeStrengthIndex.HasValue)
                    .ToList();

                var maxPrice = validData.Max(d => d.ClosePrice);

                // Create enhanced data matching the training schema
                var enhancedData = validData.Select(d => new EnhancedCryptoData
                {
                    Symbol = d.Symbol,
                    Price = d.ClosePrice,
                    LogPrice = (float)Math.Log(Math.Max(1, d.ClosePrice)),
                    PriceRatio = d.ClosePrice / maxPrice,
                    PriceSquared = d.ClosePrice * d.ClosePrice,
                    MovingAverage5Day = d.MovingAverage5Day.Value,
                    MovingAverage20Day = d.MovingAverage20Day.Value,
                    RelativeStrengthIndex = d.RelativeStrengthIndex.Value
                }).ToList();

                // Load enhanced data into IDataView
                var dataView = context.Data.LoadFromEnumerable(enhancedData);

                // Generate predictions
                var predictions = model.Transform(dataView);

                // Extract predictions and actual prices
                var predictionColumn = predictions.Schema["Score"];
                var labelColumn = predictions.Schema["Price"];

                using var cursor = predictions.GetRowCursor(new[] { predictionColumn, labelColumn });

                var labelGetter = cursor.GetGetter<float>(labelColumn);
                var predictionGetter = cursor.GetGetter<float>(predictionColumn);

                while (cursor.MoveNext())
                {
                    float actual = 0;
                    float predicted = 0;

                    labelGetter(ref actual);
                    predictionGetter(ref predicted);

                    residuals.Add(new ResidualData
                    {
                        Price = actual,
                        PredictedPrice = predicted
                    });
                }

                Console.WriteLine($"Generated {residuals.Count} residual data points for visualization");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error generating residuals: {ex.Message}");
                if (ex.InnerException != null)
                {
                    Console.WriteLine($"Inner exception: {ex.InnerException.Message}");
                }
            }

            return residuals;
        }

        private static List<CryptoData> CleanCryptoData(List<CryptoData> data)
        {
            if (data == null || !data.Any())
                return new List<CryptoData>();

            Console.WriteLine($"Cleaning data. Original count: {data.Count}");
            
            // Remove entries with zero or negative prices
            var cleanedData = data.Where(d => d.Price > 0).ToList();
            
            // Remove extreme outliers (using IQR method)
            if (cleanedData.Count > 10) // Only if we have enough data points
            {
                var prices = cleanedData.Select(d => d.Price).OrderBy(p => p).ToList();
                int q1Index = (int)(prices.Count * 0.25);
                int q3Index = (int)(prices.Count * 0.75);
                
                float q1 = prices[q1Index];
                float q3 = prices[q3Index];
                float iqr = q3 - q1;
                
                // Define outlier bounds (1.5 is the standard multiplier for outliers)
                float lowerBound = q1 - (1.5f * iqr);
                float upperBound = q3 + (1.5f * iqr);
                
                // Filter out outliers
                cleanedData = cleanedData.Where(d => d.Price >= lowerBound && d.Price <= upperBound).ToList();
            }
            
            Console.WriteLine($"Data cleaning complete. New count: {cleanedData.Count}");
            
            return cleanedData;
        }
    }

    public class CryptoData
    {
        public string Symbol { get; set; } = string.Empty;
        public float Price { get; set; }
    }

    public class CryptoDataMap : ClassMap<CryptoData>
    {
        public CryptoDataMap()
        {
            Map(m => m.Symbol).Name("Symbol");
            Map(m => m.Price).Name("Price")
                .TypeConverterOption.CultureInfo(CultureInfo.InvariantCulture);
        }
    }

    public class CryptoPrediction
    {
        [ColumnName("Score")]
        public float PredictedPrice { get; set; }
    }

    public class ResidualData
    {
        public float Price { get; set; }
        public float PredictedPrice { get; set; }
        public float Residual => Price - PredictedPrice;
    }

    public class CryptoTimeSeriesDataMap : ClassMap<CryptoTimeSeriesData>
    {
        public CryptoTimeSeriesDataMap()
        {
            Map(m => m.Symbol).Name("symbol");
            Map(m => m.Timestamp).Name("timestamp")
                .TypeConverterOption.Format("yyyy-MM-dd HH:mm:ss");
            Map(m => m.OpenPrice).Name("open")
                .TypeConverterOption.CultureInfo(CultureInfo.InvariantCulture);
            Map(m => m.HighPrice).Name("high")
                .TypeConverterOption.CultureInfo(CultureInfo.InvariantCulture);
            Map(m => m.LowPrice).Name("low")
                .TypeConverterOption.CultureInfo(CultureInfo.InvariantCulture);
            Map(m => m.ClosePrice).Name("close")
                .TypeConverterOption.CultureInfo(CultureInfo.InvariantCulture);
            Map(m => m.Volume).Name("volume")
                .TypeConverterOption.CultureInfo(CultureInfo.InvariantCulture);
        }
    }

    public class EnhancedCryptoData
    {
        public string Symbol { get; set; } = string.Empty;
        public float Price { get; set; }
        public float LogPrice { get; set; }
        public float PriceRatio { get; set; }
        public float PriceSquared { get; set; }
        public float MovingAverage5Day { get; set; }
        public float MovingAverage20Day { get; set; }
        public float RelativeStrengthIndex { get; set; }
    }
}

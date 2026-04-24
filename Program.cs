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
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using Microsoft.ML.Trainers;

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
        private const string BITCOIN_SYMBOL = "BTC";
        private const string BITCOIN_COINGECKO_ID = "bitcoin";
        private const string BITCOIN_BINANCE_SYMBOL = "BTCUSDT";
        private static bool OfflineMode =>
            string.Equals(Environment.GetEnvironmentVariable("DEEP_OFFLINE"), "1", StringComparison.OrdinalIgnoreCase)
            || string.Equals(Environment.GetEnvironmentVariable("DEEP_OFFLINE"), "true", StringComparison.OrdinalIgnoreCase);

        public static async Task Main(string[] args)
        {
            string symbol = DEFAULT_SYMBOL;
            string coinGeckoId = DEFAULT_COINGECKO_ID;
            string binanceSymbol = DEFAULT_BINANCE_SYMBOL;
            string assetName = GetAssetName(symbol);

            var assetArg = ParseAssetArgument(args);
            if (!string.IsNullOrEmpty(assetArg))
            {
                if (string.Equals(assetArg, "btc", StringComparison.OrdinalIgnoreCase)
                    || string.Equals(assetArg, "bitcoin", StringComparison.OrdinalIgnoreCase))
                {
                    symbol = BITCOIN_SYMBOL;
                    assetName = GetAssetName(symbol);
                    coinGeckoId = BITCOIN_COINGECKO_ID;
                    binanceSymbol = BITCOIN_BINANCE_SYMBOL;
                }
                else if (string.Equals(assetArg, "eth", StringComparison.OrdinalIgnoreCase)
                    || string.Equals(assetArg, "ethereum", StringComparison.OrdinalIgnoreCase))
                {
                    // Explicit Ethereum selection uses defaults.
                    symbol = DEFAULT_SYMBOL;
                    assetName = GetAssetName(symbol);
                    coinGeckoId = DEFAULT_COINGECKO_ID;
                    binanceSymbol = DEFAULT_BINANCE_SYMBOL;
                }
                else
                {
                    Console.WriteLine($"Unrecognized asset '{assetArg}'. Supported values: eth, ethereum, btc, bitcoin.");
                    return;
                }
            }
            else
            {
                Console.WriteLine("Choose asset to analyze:");
                Console.WriteLine("  1) Ethereum (default)");
                Console.WriteLine("  2) Bitcoin");
                Console.Write("Enter selection [1/2]: ");
                var choice = Console.ReadLine()?.Trim();

                if (string.Equals(choice, "2", StringComparison.OrdinalIgnoreCase)
                    || string.Equals(choice, "bitcoin", StringComparison.OrdinalIgnoreCase)
                    || string.Equals(choice, "btc", StringComparison.OrdinalIgnoreCase))
                {
                    symbol = BITCOIN_SYMBOL;
                    assetName = GetAssetName(symbol);
                    coinGeckoId = BITCOIN_COINGECKO_ID;
                    binanceSymbol = BITCOIN_BINANCE_SYMBOL;
                }
            }

            try
            {
                // Create output directory if it doesn't exist
                if (!Directory.Exists(OUTPUT_DIR))
                    Directory.CreateDirectory(OUTPUT_DIR);

                Console.WriteLine("=== Crypto Price Predictor ===");

                // Load and prepare regular data from CoinGecko
                Console.WriteLine($"Loading basic price data for {assetName}...");
                var cryptoData = await LoadDataAsync(coinGeckoId, binanceSymbol, symbol);
                if (cryptoData?.Any() != true)
                {
                    Console.WriteLine("Error: No data found for the selected asset.");
                    return;
                }
                

                // Load time series data
                Console.WriteLine($"Loading time series data for {assetName}...");
                var timeSeriesData = await LoadTimeSeriesDataAsync(coinGeckoId, binanceSymbol, symbol);
                if (timeSeriesData?.Any() == true)
                {
                    Console.WriteLine($"Loaded {timeSeriesData.Count} bars. Range: {timeSeriesData.Min(x=>x.Timestamp):u} to {timeSeriesData.Max(x=>x.Timestamp):u}");

                    // Persist snapshots for offline runs (root CSVs and timestamped copies)
                    try
                    {
                        SaveSimplePriceCsv(cryptoData, CSV_FILE_PATH);
                        SaveTimeSeriesCsv(timeSeriesData, TIMESERIES_CSV_PATH);
                        var ts = DateTime.Now.ToString("yyyyMMdd_HHmmss");
                        SaveSimplePriceCsv(cryptoData, Path.Combine(OUTPUT_DIR, $"crypto_{ts}.csv"));
                        SaveTimeSeriesCsv(timeSeriesData, Path.Combine(OUTPUT_DIR, $"crypto_timeseries_{ts}.csv"));
                        SaveSnapshotManifest(CSV_FILE_PATH, TIMESERIES_CSV_PATH, symbol);
                    }
                    catch (Exception snapEx)
                    {
                        Console.WriteLine($"Warning: Failed to save snapshots: {snapEx.Message}");
                    }
                    

                    // Rest of your code stays the same
                    Console.WriteLine("Enriching time series data with technical indicators...");
                    var enrichedData = DataPreprocessor.EnrichTimeSeriesData(timeSeriesData);

                    // Require enough enriched rows with indicators
                    if (!enrichedData.Any(d => d.MovingAverage5Day.HasValue && d.MovingAverage20Day.HasValue && d.RelativeStrengthIndex.HasValue))
                    {
                        Console.WriteLine("Not enough enriched data with indicators to train a model.");
                        return;
                    }

                    // Train the model using enriched data
                    var model = TrainModel(enrichedData);

                    Console.WriteLine($"Enrichment complete: {enrichedData.Count} records processed");

                    // Create visualizations for the first symbol in the dataset
                    if (enrichedData.Any())
                    {
                        Console.WriteLine($"Creating time series visualization for {assetName}...");
                        CreateTimeSeriesVisualization(enrichedData, symbol);
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
                            symbol,
                            predictionVsActualPath
                        );

                        Console.WriteLine($"Prediction vs actual chart saved to: {Path.GetFullPath(predictionVsActualPath)}");
                    }

                    // Use the model for predictions (example)
                    var latestIndicators = enrichedData.Last(); // latest enriched data point
                    var maxPrice = enrichedData.Count > 0 ? enrichedData.Max(x => x.ClosePrice) : 1f;

                    var prediction = Predict(model, new EnhancedCryptoData
                    {
                        Symbol = symbol,
                        Price = latestIndicators.ClosePrice,
                        LogPrice = (float)Math.Log(Math.Max(1, latestIndicators.ClosePrice)),
                        PriceRatio = latestIndicators.ClosePrice / maxPrice,
                        PriceSquared = latestIndicators.ClosePrice * latestIndicators.ClosePrice,
                        PriceChangePct = latestIndicators.PriceChangePercentage.GetValueOrDefault(),
                        MovingAverage5Day = latestIndicators.MovingAverage5Day ?? 0f,
                        MovingAverage20Day = latestIndicators.MovingAverage20Day ?? 0f,
                        RelativeStrengthIndex = latestIndicators.RelativeStrengthIndex ?? 50f,
                        Volatility7 = latestIndicators.Volatility7 ?? 0f,
                        Volatility14 = latestIndicators.Volatility14 ?? 0f,
                        Volatility21 = latestIndicators.Volatility21 ?? 0f,
                        Volatility7Z = latestIndicators.Volatility7Z ?? 0f,
                        Volatility14Z = latestIndicators.Volatility14Z ?? 0f,
                        Volatility21Z = latestIndicators.Volatility21Z ?? 0f
                    });

                    Console.WriteLine($"Predicted Best Price to Buy {symbol}: {prediction.PredictedPrice:C2}");
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

        private static async Task<List<CryptoData>> LoadDataAsync(string coinGeckoId, string binanceSymbol, string selectedSymbol)
        {
            try
            {
                if (!OfflineMode)
                {
                    var coinGeckoDataFetcher = new CoinGeckoDataFetcher();
                    var coinGeckoData = await coinGeckoDataFetcher.GetPricesAsync(coinGeckoId, 365);
                    if (coinGeckoData?.Any() == true)
                    {
                        NormalizeSymbols(coinGeckoData, selectedSymbol);
                        Console.WriteLine("Successfully loaded data from CoinGecko!");
                        Console.WriteLine($"Loaded {coinGeckoData.Count} price records from CoinGecko");
                        return CleanCryptoData(coinGeckoData);
                    }
                    Console.WriteLine("No data returned from CoinGecko, trying Binance...");

                    var binanceDataFetcher = new BinanceDataFetcher();
                    var data = await binanceDataFetcher.GetPricesAsync(binanceSymbol, 365);
                    if (data?.Any() == true)
                    {
                        NormalizeSymbols(data, selectedSymbol);
                        Console.WriteLine("Successfully loaded data from Binance!");
                        Console.WriteLine($"Loaded {data.Count} price records from Binance");
                        return CleanCryptoData(data);
                    }
                }

                Console.WriteLine("Both online APIs failed, falling back to CSV data...");
                var csvData = LoadData();
                csvData = csvData.Where(d => string.Equals(d.Symbol, selectedSymbol, StringComparison.OrdinalIgnoreCase)).ToList();
                Console.WriteLine($"Loaded {csvData.Count} price records from CSV");
                return CleanCryptoData(csvData);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error loading online data: {ex.Message}");
                Console.WriteLine("Falling back to CSV data...");
                var csvData = LoadData();
                Console.WriteLine($"Loaded {csvData.Count} price records from CSV");
                return CleanCryptoData(csvData);
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
                var records = csv.GetRecords<CryptoData>().ToList();
                foreach (var record in records)
                {
                    record.Symbol = NormalizeSymbolName(record.Symbol);
                }
                return records;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error loading data: {ex.Message}");
                throw; // Rethrow to be handled by the main error handler
            }
        }

        private static async Task<List<CryptoTimeSeriesData>> LoadTimeSeriesDataAsync(string coinGeckoId, string binanceSymbol, string selectedSymbol)
        {
            try
            {
                if (!OfflineMode)
                {
                    // First try Binance
                    var binanceDataFetcher = new BinanceDataFetcher();
                    var data = await binanceDataFetcher.GetOHLCVAsync(binanceSymbol, "1d", 365);
                    if (data?.Any() == true)
                    {
                        NormalizeSymbols(data, selectedSymbol);
                        Console.WriteLine("Successfully loaded time series data from Binance!");
                        Console.WriteLine($"Loaded {data.Count} time series records from Binance");
                        return data;
                    }
                    Console.WriteLine("No time series data returned from Binance, trying CoinGecko...");

                    // Then try CoinGecko as backup
                    var coinGeckoDataFetcher = new CoinGeckoDataFetcher();
                    var coinGeckoData = await coinGeckoDataFetcher.GetOHLCVAsync(coinGeckoId, 365);
                    if (coinGeckoData?.Any() == true)
                    {
                        NormalizeSymbols(coinGeckoData, selectedSymbol);
                        Console.WriteLine("Successfully loaded time series data from CoinGecko!");
                        Console.WriteLine($"Loaded {coinGeckoData.Count} time series records from CoinGecko");
                        return coinGeckoData;
                    }
                }

                // If both fail, fall back to CSV
                Console.WriteLine("Both online APIs failed, falling back to CSV time series data...");
                var csvData = LoadTimeSeriesData();
                csvData = csvData.Where(d => string.Equals(d.Symbol, selectedSymbol, StringComparison.OrdinalIgnoreCase)).ToList();
                Console.WriteLine($"Loaded {csvData.Count} time series records from CSV");
                return csvData;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error loading online time series data: {ex.Message}");
                Console.WriteLine("Falling back to CSV time series data...");
                var csvData = LoadTimeSeriesData();
                Console.WriteLine($"Loaded {csvData.Count} time series records from CSV");
                return csvData;
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

                var records = csv.GetRecords<CryptoTimeSeriesData>().ToList();
                foreach (var record in records)
                {
                    record.Symbol = NormalizeSymbolName(record.Symbol);
                }
                return records;
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
                // Build next-day label and features per symbol; split chronologically
                var eligible = enrichedData
                    .Where(d => d.MovingAverage5Day.HasValue && d.MovingAverage20Day.HasValue && d.RelativeStrengthIndex.HasValue)
                    .OrderBy(d => d.Symbol)
                    .ThenBy(d => d.Timestamp)
                    .ToList();

                var shifted = new List<EnhancedCryptoData>();
                var bySymbol = eligible.GroupBy(x => x.Symbol);
                foreach (var g in bySymbol)
                {
                    var rows = g.OrderBy(x => x.Timestamp).ToList();
                    for (int i = 0; i < rows.Count - 1; i++)
                    {
                        var cur = rows[i];
                        var nxt = rows[i + 1];
                        shifted.Add(new EnhancedCryptoData
                        {
                            Symbol = cur.Symbol,
                            // Price becomes next day's close (label)
                            Price = nxt.ClosePrice,
                            LogPrice = (float)Math.Log(Math.Max(1, cur.ClosePrice)),
                            PriceRatio = 0f, // placeholder not used in features
                            PriceSquared = cur.ClosePrice * cur.ClosePrice,
                            // recently observed return can carry momentum information
                            PriceChangePct = cur.PriceChangePercentage.GetValueOrDefault(),
                            MovingAverage5Day = cur.MovingAverage5Day.GetValueOrDefault(),
                            MovingAverage20Day = cur.MovingAverage20Day.GetValueOrDefault(),
                            RelativeStrengthIndex = cur.RelativeStrengthIndex.GetValueOrDefault(),
                            Volatility7 = cur.Volatility7.GetValueOrDefault(),
                            Volatility14 = cur.Volatility14.GetValueOrDefault(),
                            Volatility21 = cur.Volatility21.GetValueOrDefault(),
                            Volatility7Z = cur.Volatility7Z.GetValueOrDefault(),
                            Volatility14Z = cur.Volatility14Z.GetValueOrDefault(),
                            Volatility21Z = cur.Volatility21Z.GetValueOrDefault()
                        });
                    }
                }

                if (shifted.Count < 50)
                    throw new InvalidOperationException("Not enough shifted rows to train (need at least 50).");

                // Determine whether we actually have multiple symbols; if not, skip one-hot to avoid constant features
                var distinctSymbolCount = shifted.Select(s => s.Symbol).Distinct(StringComparer.OrdinalIgnoreCase).Count();

                // Chronological split 80/20
                var ordered = shifted.OrderBy(x => x.Symbol).ThenBy(x => x.LogPrice).ToList();
                // Better ordering by implicit time: reconstruct via grouping retained above; here keep original order
                ordered = shifted; // already chronological per-symbol appended
                int splitIdx = (int)(ordered.Count * 0.8);
                var trainingData = ordered.Take(splitIdx).ToList();
                var testData = ordered.Skip(splitIdx).ToList();

                var dataView = context.Data.LoadFromEnumerable(trainingData);
                var testDataView = context.Data.LoadFromEnumerable(testData);

                // Pipeline with technical indicators only (avoid target leakage from same-bar price transforms)
                IEstimator<ITransformer> pipeline = context.Transforms.ReplaceMissingValues(new[] {
                        new InputOutputColumnPair("MovingAverage5Day"),
                        new InputOutputColumnPair("MovingAverage20Day"),
                        new InputOutputColumnPair("RelativeStrengthIndex"),
                        new InputOutputColumnPair("PriceChangePct"),
                        new InputOutputColumnPair("Volatility7"),
                        new InputOutputColumnPair("Volatility14"),
                        new InputOutputColumnPair("Volatility21"),
                        new InputOutputColumnPair("Volatility7Z"),
                        new InputOutputColumnPair("Volatility14Z"),
                        new InputOutputColumnPair("Volatility21Z")
                    })
                    .Append(context.Transforms.Concatenate("NumFeatures",
                        "MovingAverage5Day", "MovingAverage20Day", "RelativeStrengthIndex", "PriceChangePct", "Volatility7", "Volatility14", "Volatility21", "Volatility7Z", "Volatility14Z", "Volatility21Z"));

                if (distinctSymbolCount > 1)
                {
                    pipeline = pipeline
                        .Append(context.Transforms.Categorical.OneHotEncoding("SymbolEncoded", "Symbol"))
                        .Append(context.Transforms.Concatenate("Features", "NumFeatures", "SymbolEncoded"));
                }
                else
                {
                    pipeline = pipeline
                        .Append(context.Transforms.CopyColumns("Features", "NumFeatures"));
                }

                pipeline = pipeline
                    .Append(context.Transforms.NormalizeMinMax("Features"))
                    .Append(context.Regression.Trainers.Sdca(
                        labelColumnName: "Price",
                        featureColumnName: "Features",
                        maximumNumberOfIterations: 100));

                Console.WriteLine("Starting model training...");
                var model = pipeline.Fit(dataView);
                Console.WriteLine("Model training completed successfully");

                // Evaluate on test set
                var predictions = model.Transform(testDataView);
                var metrics = context.Regression.Evaluate(predictions, labelColumnName: "Price", scoreColumnName: "Score");
                Console.WriteLine($"Evaluation — RMSE: {metrics.RootMeanSquaredError:F4}, MAE: {metrics.MeanAbsoluteError:F4}, R^2: {metrics.RSquared:F4}");

                // Optional: permutation feature importance (returns importance dictionary)
                var featureImportances = PrintFeatureImportance(context, model, testDataView);

                // Compute a model-informed volatility index per symbol (using PFI-derived weights)
                try
                {
                    var volIndexSeries = ComputeAndPrintVolatilityIndex(enrichedData, featureImportances);

                    // Plot volatility series and volatility index per symbol
                    foreach (var kv in volIndexSeries)
                    {
                        var symbol = kv.Key;
                        var series = kv.Value;
                        string ts = DateTime.Now.ToString("yyyyMMdd_HHmmss");

                        var volPath = Path.Combine(OUTPUT_DIR, $"{symbol}_volatility_{ts}.png");
                        Visualization.CreateVolatilityChart(enrichedData, symbol, volPath);
                        Console.WriteLine($"Volatility chart saved to: {Path.GetFullPath(volPath)}");

                        var idxPath = Path.Combine(OUTPUT_DIR, $"{symbol}_volatility_index_{ts}.png");
                        Visualization.CreateVolatilityIndexChart(series, symbol, idxPath);
                        Console.WriteLine($"Volatility index chart saved to: {Path.GetFullPath(idxPath)}");
                    }
                }
                catch (Exception viEx)
                {
                    Console.WriteLine($"Volatility index warning: {viEx.Message}");
                }

                // Quick experiment: compare models using different volatility windows
                try
                {
                    EvaluateVolatilityVariants(context, trainingData, testData);
                }
                catch (Exception evEx)
                {
                    Console.WriteLine($"Volatility experiment warning: {evEx.Message}");
                }

                // Walk-forward evaluation (3 folds)
                try
                {
                    WalkForwardEvaluate(context, trainingData, folds: 3);
                }
                catch (Exception wfEx)
                {
                    Console.WriteLine($"Walk-forward evaluation warning: {wfEx.Message}");
                }

                // Save model for reuse
                try
                {
                    if (!Directory.Exists(OUTPUT_DIR)) Directory.CreateDirectory(OUTPUT_DIR);
                    var modelPath = Path.Combine(OUTPUT_DIR, $"model_{DateTime.Now:yyyyMMdd_HHmmss}.zip");
                    context.Model.Save(model, dataView.Schema, modelPath);
                    Console.WriteLine($"Model saved to: {Path.GetFullPath(modelPath)}");
                }
                catch (Exception saveEx)
                {
                    Console.WriteLine($"Warning: Failed to save model: {saveEx.Message}");
                }

                return model;
            }
            catch (Exception ex)
            {
                throw new InvalidOperationException($"Model training failed: {ex.Message}", ex);
            }
        }

        private static void EvaluateVolatilityVariants(MLContext context, List<EnhancedCryptoData> train, List<EnhancedCryptoData> test)
        {
            var variants = new List<string[]>
            {
                new[] { "Volatility7" },
                new[] { "Volatility14" },
                new[] { "Volatility21" },
                new[] { "Volatility7", "Volatility14", "Volatility21" }
            };

            Console.WriteLine("\nVolatility window experiments:");
            foreach (var v in variants)
            {
                var featureList = new List<string> { "MovingAverage5Day", "MovingAverage20Day", "RelativeStrengthIndex", "PriceChangePct" };
                foreach (var baseName in v)
                {
                    featureList.Add(baseName);
                    featureList.Add(baseName + "Z");
                }

                var pipeline = context.Transforms.ReplaceMissingValues(featureList.Select(f => new InputOutputColumnPair(f)).ToArray())
                    .Append(context.Transforms.Concatenate("Features", featureList.ToArray()))
                    .Append(context.Transforms.NormalizeMinMax("Features"))
                    .Append(context.Regression.Trainers.Sdca(labelColumnName: "Price", featureColumnName: "Features", maximumNumberOfIterations: 100));

                var model = pipeline.Fit(context.Data.LoadFromEnumerable(train));
                var preds = model.Transform(context.Data.LoadFromEnumerable(test));
                var m = context.Regression.Evaluate(preds, labelColumnName: "Price", scoreColumnName: "Score");

                Console.WriteLine($" Variant [{string.Join(',', v)}] => RMSE={m.RootMeanSquaredError:F4}, MAE={m.MeanAbsoluteError:F4}, R^2={m.RSquared:F4}");
            }
        }

        private static IDictionary<string,double> PrintFeatureImportance(MLContext context, ITransformer model, IDataView data)
        {
            var result = new Dictionary<string, double>(StringComparer.OrdinalIgnoreCase);
            try
            {
                // Prefer calling PFI with the strongly-typed regression transformer so it
                // uses the trainer's configured feature column name.
                RegressionPredictionTransformer<LinearRegressionModelParameters>? reg = null;
                if (model is TransformerChain<ITransformer> chain)
                {
                    foreach (var tr in chain)
                    {
                        if (tr is RegressionPredictionTransformer<LinearRegressionModelParameters> r)
                            reg = r;
                    }
                }
                else if (model is RegressionPredictionTransformer<LinearRegressionModelParameters> r)
                {
                    reg = r;
                }

                if (reg != null)
                {
                    // Get slot names for pretty-printing
                    var preview = reg.Transform(data);
                    var schema = preview.Schema;
                    var featureNames = new List<string>();
                    if (schema.GetColumnOrNull("Features") != null)
                    {
                        try
                        {
                            var featureCol = schema["Features"];
                            VBuffer<ReadOnlyMemory<char>> slotNames = default;
                            featureCol.GetSlotNames(ref slotNames);
                            if (slotNames.Length > 0)
                                featureNames = slotNames.DenseValues().Select(s => s.ToString()).ToList();
                        }
                        catch { }
                    }

                    // Use the transformed preview (has the 'Features' column) when calling PFI
                    var transformedPreview = reg.Transform(data);

                    var pfi = context.Regression.PermutationFeatureImportance(
                        reg,
                        transformedPreview,
                        labelColumnName: "Price");

                    // Align names/count
                    if (featureNames.Count == 0)
                        featureNames = Enumerable.Range(0, pfi.Length).Select(i => $"Feature[{i}]").ToList();

                    Console.WriteLine("\nFeature Importance:");
                    for (int i = 0; i < pfi.Length && i < featureNames.Count; i++)
                    {
                        var mean = pfi[i].RootMeanSquaredError.Mean;
                        if (double.IsNaN(mean)) mean = 0;
                        Console.WriteLine($"Feature {featureNames[i]}: {mean:F4}");
                        result[featureNames[i]] = mean;
                    }
                }
                else
                {
                    // Fallback to the generic overload (older ML.NET) and print what we can
                    var preview = model.Transform(data);
                    var schema = preview.Schema;

                    // Call PFI on the transformed preview where 'Features' exists
                    var pfiDict = context.Regression.PermutationFeatureImportance(
                        model,
                        preview,
                        labelColumnName: "Price");

                    Console.WriteLine("\nFeature Importance:");
                    foreach (var kvp in pfiDict.OrderByDescending(k => k.Value.RootMeanSquaredError.Mean))
                    {
                        var mean = kvp.Value.RootMeanSquaredError.Mean;
                        if (double.IsNaN(mean)) mean = 0;
                        Console.WriteLine($"Feature {kvp.Key}: {mean:F4}");
                        result[kvp.Key] = mean;
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Feature importance calculation error: {ex.Message}");
                try
                {
                    Console.WriteLine("\nAvailable columns in schema:");
                    foreach (var column in data.Schema)
                        Console.WriteLine($"- {column.Name} ({column.Type})");
                }
                catch { }
                if (ex.InnerException != null)
                    Console.WriteLine($"Inner exception: {ex.InnerException.Message}");
            }

            return result;
        }

        private static Dictionary<string, List<(DateTime Timestamp, double Index)>> ComputeAndPrintVolatilityIndex(List<CryptoTimeSeriesData> enrichedData, IDictionary<string,double> featureImportances)
        {
            var seriesBySymbol = new Dictionary<string, List<(DateTime Timestamp, double Index)>>();

            // Volatility feature names we care about
            var volNames = new[] { "Volatility7", "Volatility14", "Volatility21" };

            // Get weights from feature importances (use absolute values)
            var weights = new double[volNames.Length];
            for (int i = 0; i < volNames.Length; i++)
            {
                var key = featureImportances.Keys.FirstOrDefault(k => k.Equals(volNames[i], StringComparison.OrdinalIgnoreCase) || k.IndexOf(volNames[i], StringComparison.OrdinalIgnoreCase) >= 0);
                if (key != null && featureImportances.TryGetValue(key, out var v))
                    weights[i] = Math.Abs(v);
                else
                    weights[i] = 0.0;
            }

            // If all weights are zero, fall back to equal weights
            if (weights.All(w => w == 0))
            {
                for (int i = 0; i < weights.Length; i++) weights[i] = 1.0;
            }

            // Normalize weights
            var sumW = weights.Sum();
            if (sumW == 0) sumW = 1.0;
            for (int i = 0; i < weights.Length; i++) weights[i] /= sumW;

            // Compute index per symbol and collect time series
            foreach (var g in enrichedData.GroupBy(d => d.Symbol))
            {
                var symbol = g.Key;
                var rows = g.OrderBy(d => d.Timestamp).ToList();

                var scores = new List<double>();
                foreach (var r in rows)
                {
                    var v7 = r.Volatility7 ?? 0f;
                    var v14 = r.Volatility14 ?? 0f;
                    var v21 = r.Volatility21 ?? 0f;
                    var score = weights[0] * v7 + weights[1] * v14 + weights[2] * v21;
                    scores.Add(score);
                }

                if (!scores.Any()) continue;

                var sorted = scores.OrderBy(s => s).ToList();
                var tsPairs = new List<(DateTime Timestamp, double Index)>();

                for (int i = 0; i < rows.Count; i++)
                {
                    var score = scores[i];
                    int less = sorted.TakeWhile(s => s < score).Count();
                    int equal = sorted.Count(s => s == score);
                    double percentile = (less + 0.5 * equal) / sorted.Count;
                    tsPairs.Add((rows[i].Timestamp, percentile * 100.0));
                }

                // Compute percentile of latest score
                var latestScore = scores.Last();
                int latestLess = sorted.TakeWhile(s => s < latestScore).Count();
                int latestEqual = sorted.Count(s => s == latestScore);
                double latestPercentile = (latestLess + 0.5 * latestEqual) / sorted.Count;
                double latestIndex = latestPercentile * 100.0;

                string band = latestIndex < 33 ? "Low" : (latestIndex < 66 ? "Medium" : "High");

                Console.WriteLine($"Volatility Index for {symbol}: {latestIndex:F1} ({band}) — weighted score: {latestScore:F4}");

                seriesBySymbol[symbol] = tsPairs;
            }

            return seriesBySymbol;
        }

        private static string ParseAssetArgument(string[] args)
        {
            if (args == null || args.Length == 0)
                return string.Empty;

            for (int i = 0; i < args.Length; i++)
            {
                var arg = args[i]?.Trim();
                if (string.IsNullOrEmpty(arg))
                    continue;

                if (arg.Equals("-asset", StringComparison.OrdinalIgnoreCase)
                    || arg.Equals("--asset", StringComparison.OrdinalIgnoreCase))
                {
                    if (i + 1 < args.Length)
                        return args[i + 1].Trim();
                }

                if (arg.StartsWith("-asset=", StringComparison.OrdinalIgnoreCase)
                    || arg.StartsWith("--asset=", StringComparison.OrdinalIgnoreCase))
                {
                    var parts = arg.Split(new[] { '=' }, 2);
                    if (parts.Length == 2)
                        return parts[1].Trim();
                }
            }

            return string.Empty;
        }

        private static void NormalizeSymbols(List<CryptoData> data, string symbol)
        {
            if (data == null || string.IsNullOrWhiteSpace(symbol))
                return;

            foreach (var item in data)
                item.Symbol = symbol;
        }

        private static void NormalizeSymbols(List<CryptoTimeSeriesData> data, string symbol)
        {
            if (data == null || string.IsNullOrWhiteSpace(symbol))
                return;

            foreach (var item in data)
                item.Symbol = symbol;
        }

        private static string GetAssetName(string symbol)
        {
            return symbol?.ToUpperInvariant() switch
            {
                "BTC" => "Bitcoin",
                "ETH" => "Ethereum",
                _ => symbol ?? string.Empty,
            };
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
                // Ensure we have valid technical indicators and shift label by +1
                var valid = enrichedData
                    .Where(d => d.MovingAverage5Day.HasValue && d.MovingAverage20Day.HasValue && d.RelativeStrengthIndex.HasValue)
                    .OrderBy(d => d.Symbol)
                    .ThenBy(d => d.Timestamp)
                    .ToList();

                var enhancedData = new List<EnhancedCryptoData>();
                foreach (var g in valid.GroupBy(v => v.Symbol))
                {
                    var rows = g.OrderBy(x => x.Timestamp).ToList();
                    for (int i = 0; i < rows.Count - 1; i++)
                    {
                        var cur = rows[i];
                        var nxt = rows[i + 1];
                        enhancedData.Add(new EnhancedCryptoData
                        {
                            Symbol = cur.Symbol,
                            Price = nxt.ClosePrice,
                            LogPrice = (float)Math.Log(Math.Max(1, cur.ClosePrice)),
                            PriceRatio = 0f,
                            PriceSquared = cur.ClosePrice * cur.ClosePrice,
                            PriceChangePct = cur.PriceChangePercentage.GetValueOrDefault(),
                            MovingAverage5Day = cur.MovingAverage5Day.GetValueOrDefault(),
                            MovingAverage20Day = cur.MovingAverage20Day.GetValueOrDefault(),
                            RelativeStrengthIndex = cur.RelativeStrengthIndex.GetValueOrDefault(),
                            Volatility7 = cur.Volatility7.GetValueOrDefault(),
                            Volatility14 = cur.Volatility14.GetValueOrDefault(),
                            Volatility21 = cur.Volatility21.GetValueOrDefault(),
                            Volatility7Z = cur.Volatility7Z.GetValueOrDefault(),
                            Volatility14Z = cur.Volatility14Z.GetValueOrDefault(),
                            Volatility21Z = cur.Volatility21Z.GetValueOrDefault()
                        });
                    }
                }

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

        private static void WalkForwardEvaluate(MLContext context, List<EnhancedCryptoData> orderedTrainingData, int folds)
        {
            if (orderedTrainingData.Count < folds * 20) return; // need enough data

            Console.WriteLine($"Walk-forward evaluation ({folds} folds):");
            var foldMetrics = new List<(double rmse, double mae, double r2)>();


            int n = orderedTrainingData.Count;
            for (int f = 1; f <= folds; f++)
            {
                int trainEnd = (int)(n * (0.5 + 0.1 * f)); // 60%, 70%, 80%
                if (trainEnd >= n - 5) break;
                int testEnd = Math.Min(n, trainEnd + Math.Max(5, (int)(n * 0.1)));
                var train = orderedTrainingData.Take(trainEnd).ToList();
                var test = orderedTrainingData.Skip(trainEnd).Take(testEnd - trainEnd).ToList();

                var distinctSymbolCount = train.Select(s => s.Symbol).Distinct(StringComparer.OrdinalIgnoreCase).Count();

                IEstimator<ITransformer> pipeline = context.Transforms.ReplaceMissingValues(new[] {
                        new InputOutputColumnPair("MovingAverage5Day"),
                        new InputOutputColumnPair("MovingAverage20Day"),
                        new InputOutputColumnPair("RelativeStrengthIndex"),
                        new InputOutputColumnPair("PriceChangePct"),
                        new InputOutputColumnPair("Volatility7"),
                        new InputOutputColumnPair("Volatility14"),
                        new InputOutputColumnPair("Volatility21"),
                        new InputOutputColumnPair("Volatility7Z"),
                        new InputOutputColumnPair("Volatility14Z"),
                        new InputOutputColumnPair("Volatility21Z")
                    })
                    .Append(context.Transforms.Concatenate("NumFeatures",
                        "MovingAverage5Day", "MovingAverage20Day", "RelativeStrengthIndex", "PriceChangePct", "Volatility7", "Volatility14", "Volatility21", "Volatility7Z", "Volatility14Z", "Volatility21Z"));

                if (distinctSymbolCount > 1)
                {
                    pipeline = pipeline
                        .Append(context.Transforms.Categorical.OneHotEncoding("SymbolEncoded", "Symbol"))
                        .Append(context.Transforms.Concatenate("Features", "NumFeatures", "SymbolEncoded"));
                }
                else
                {
                    pipeline = pipeline
                        .Append(context.Transforms.CopyColumns("Features", "NumFeatures"));
                }

                pipeline = pipeline
                    .Append(context.Transforms.NormalizeMinMax("Features"))
                    .Append(context.Regression.Trainers.Sdca(
                        labelColumnName: "Price",
                        featureColumnName: "Features",
                        maximumNumberOfIterations: 100));

                var model = pipeline.Fit(context.Data.LoadFromEnumerable(train));
                var preds = model.Transform(context.Data.LoadFromEnumerable(test));
                var m = context.Regression.Evaluate(preds, labelColumnName: "Price", scoreColumnName: "Score");
                foldMetrics.Add((m.RootMeanSquaredError, m.MeanAbsoluteError, m.RSquared));
                Console.WriteLine($"  Fold {f}: RMSE={m.RootMeanSquaredError:F4} MAE={m.MeanAbsoluteError:F4} R^2={m.RSquared:F4}");
            }

            if (foldMetrics.Count > 0)
            {
                var avgRmse = foldMetrics.Average(x => x.rmse);
                var avgMae = foldMetrics.Average(x => x.mae);
                var avgr2 = foldMetrics.Average(x => x.r2);
                Console.WriteLine($"  Avg: RMSE={avgRmse:F4} MAE={avgMae:F4} R^2={avgr2:F4}");
            }
        }

        private static void SaveSimplePriceCsv(List<CryptoData> data, string path)
        {
            var config = new CsvConfiguration(CultureInfo.InvariantCulture)
            {
                HasHeaderRecord = true
            };

            var mergedData = new List<CryptoData>();
            if (File.Exists(path))
            {
                try
                {
                    using var reader = new StreamReader(path);
                    using var csvReader = new CsvReader(reader, config);
                    csvReader.Context.RegisterClassMap<CryptoDataMap>();
                    mergedData = csvReader.GetRecords<CryptoData>().ToList();
                }
                catch
                {
                    mergedData = new List<CryptoData>();
                }
            }

            mergedData.AddRange(data);
            var distinctRows = mergedData
                .Where(row => !string.IsNullOrWhiteSpace(row.Symbol))
                .Select(row => new CryptoData { Symbol = NormalizeSymbolName(row.Symbol), Price = row.Price })
                .DistinctBy(row => (row.Symbol, row.Price))
                .ToList();

            Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(path)) ?? ".");
            using var writer = new StreamWriter(path, false, new UTF8Encoding(false));
            using var csv = new CsvWriter(writer, config);
            csv.WriteHeader<CryptoData>();
            csv.NextRecord();
            foreach (var row in distinctRows)
            {
                csv.WriteRecord(row);
                csv.NextRecord();
            }
        }

        private static void SaveTimeSeriesCsv(List<CryptoTimeSeriesData> data, string path)
        {
            var config = new CsvConfiguration(CultureInfo.InvariantCulture)
            {
                HasHeaderRecord = true
            };

            var mergedData = new Dictionary<(string Symbol, DateTime Timestamp), CryptoTimeSeriesData>();
            if (File.Exists(path))
            {
                try
                {
                    using var reader = new StreamReader(path);
                    using var csvReader = new CsvReader(reader, config);
                    csvReader.Context.RegisterClassMap<CryptoTimeSeriesDataMap>();
                    foreach (var record in csvReader.GetRecords<CryptoTimeSeriesData>())
                    {
                        var normalizedSymbol = NormalizeSymbolName(record.Symbol);
                        var key = (normalizedSymbol, record.Timestamp);
                        record.Symbol = normalizedSymbol;
                        mergedData[key] = record;
                    }
                }
                catch
                {
                    mergedData.Clear();
                }
            }

            foreach (var item in data)
            {
                var normalizedSymbol = NormalizeSymbolName(item.Symbol);
                var key = (normalizedSymbol, item.Timestamp);
                item.Symbol = normalizedSymbol;
                mergedData[key] = item;
            }

            Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(path)) ?? ".");
            using var writer = new StreamWriter(path, false, new UTF8Encoding(false));
            using var csv = new CsvWriter(writer, config);
            csv.WriteField("symbol"); csv.WriteField("timestamp"); csv.WriteField("open"); csv.WriteField("high"); csv.WriteField("low"); csv.WriteField("close"); csv.WriteField("volume");
            csv.NextRecord();
            foreach (var r in mergedData.Values.OrderBy(r => r.Symbol).ThenBy(r => r.Timestamp))
            {
                csv.WriteField(r.Symbol);
                csv.WriteField(r.Timestamp.ToString("yyyy-MM-dd HH:mm:ss", CultureInfo.InvariantCulture));
                csv.WriteField(r.OpenPrice);
                csv.WriteField(r.HighPrice);
                csv.WriteField(r.LowPrice);
                csv.WriteField(r.ClosePrice);
                csv.WriteField(r.Volume);
                csv.NextRecord();
            }
        }

        private static void SaveSnapshotManifest(string simplePath, string tsPath, string symbol)
        {
            try
            {
                var manifest = new
                {
                    createdAt = DateTime.UtcNow,
                    symbol,
                    files = new[]
                    {
                        new { path = Path.GetFullPath(simplePath), sha256 = ComputeSha256(simplePath) },
                        new { path = Path.GetFullPath(tsPath), sha256 = ComputeSha256(tsPath) }
                    }
                };
                var json = JsonSerializer.Serialize(manifest, new JsonSerializerOptions { WriteIndented = true });
                var outPath = Path.Combine(OUTPUT_DIR, $"snapshot_manifest_{DateTime.Now:yyyyMMdd_HHmmss}.json");
                Directory.CreateDirectory(OUTPUT_DIR);
                File.WriteAllText(outPath, json, new UTF8Encoding(false));
                Console.WriteLine($"Snapshot manifest saved: {Path.GetFullPath(outPath)}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Warning: Failed to write manifest: {ex.Message}");
            }
        }

        private static string ComputeSha256(string path)
        {
            if (!File.Exists(path)) return string.Empty;
            using var sha = SHA256.Create();
            using var fs = File.OpenRead(path);
            var hash = sha.ComputeHash(fs);
            return string.Concat(hash.Select(b => b.ToString("x2")));
        }

        private static string NormalizeSymbolName(string symbol)
        {
            if (string.IsNullOrWhiteSpace(symbol))
                return string.Empty;

            switch (symbol.Trim().ToUpperInvariant())
            {
                case "ETHEREUM":
                case "ETH":
                    return "ETH";
                case "BITCOIN":
                case "BTC":
                case "XBT":
                    return "BTC";
                default:
                    return symbol.Trim().ToUpperInvariant();
            }
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
            Map(m => m.Symbol).Name("Symbol", "symbol");
            Map(m => m.Price).Name("Price", "price")
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
        public float PriceChangePct { get; set; }
        public float Volatility7 { get; set; }
        public float Volatility14 { get; set; }
        public float Volatility21 { get; set; }
        public float Volatility7Z { get; set; }
        public float Volatility14Z { get; set; }
        public float Volatility21Z { get; set; }
        public float MovingAverage5Day { get; set; }
        public float MovingAverage20Day { get; set; }
        public float RelativeStrengthIndex { get; set; }
    }
}

using Microsoft.ML;
using Microsoft.ML.Data;
using CsvHelper;
using CsvHelper.Configuration;
using System.Globalization;
using System.Linq;
using System.Collections.Generic;
using System.IO;
using System;
using CryptoPredictor; // Remove the incorrect using statement


namespace CryptoPredictor
{
    public class Program
    {
        private const string CSV_FILE_PATH = "crypto.csv";
        private const string TIMESERIES_CSV_PATH = "crypto_timeseries.csv"; // For time series data
        private const string OUTPUT_DIR = "output";

        public static void Main(string[] args)
        {
            try
            {
                // Create output directory if it doesn't exist
                if (!Directory.Exists(OUTPUT_DIR))
                    Directory.CreateDirectory(OUTPUT_DIR);

                Console.WriteLine("=== Crypto Price Predictor ===");

                // Load and prepare regular data
                Console.WriteLine("Loading basic price data...");
                var cryptoData = LoadData();
                if (cryptoData?.Any() != true)
                {
                    Console.WriteLine("Error: No data found in the CSV file.");
                    return;
                }

                // Load time series data if available
                Console.WriteLine("Loading time series data...");
                var timeSeriesData = LoadTimeSeriesData();
                if (timeSeriesData?.Any() == true)
                {
                    Console.WriteLine($"Loaded {timeSeriesData.Count} time series records");

                    // Preprocess time series data
                    Console.WriteLine("Enriching time series data with technical indicators...");
                    var enrichedData = DataPreprocessor.EnrichTimeSeriesData(timeSeriesData);
                    Console.WriteLine($"Enrichment complete: {enrichedData.Count} records processed");

                    // Create visualizations for the first symbol in the dataset
                    if (enrichedData.Any())
                    {
                        var firstSymbol = enrichedData.First().Symbol;
                        Console.WriteLine($"Creating time series visualization for {firstSymbol}...");
                        CreateTimeSeriesVisualization(enrichedData, firstSymbol);
                    }
                }

                // Train the model
                Console.WriteLine("Training the price prediction model...");
                var model = TrainModel(cryptoData);

                // Generate residuals for visualization
                var residuals = GenerateResiduals(model, cryptoData);

                // Create visualization for prediction vs actual
                if (residuals.Any())
                {
                    Console.WriteLine("Creating prediction vs actual visualization...");
                    Visualization.CreatePredictionVsActualChart(
                        residuals,
                        Path.Combine(OUTPUT_DIR, "prediction_vs_actual.png")
                    );
                }

                // Use the model for predictions (example)
                var prediction = Predict(model, new CryptoData { Symbol = "BTC", Price = 50000 });
                Console.WriteLine($"Predicted Best Price to Buy BTC: {prediction.PredictedPrice:C2}");
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

        private static ITransformer TrainModel(IList<CryptoData> data)
        {
            var context = new MLContext(seed: 42);

            try
            {
                var dataView = context.Data.LoadFromEnumerable(data);

                // Modify pipeline to handle potential numeric overflow
                var pipeline = context.Transforms.Categorical.OneHotEncoding(
                        outputColumnName: "SymbolEncoded",
                        inputColumnName: "Symbol")
                    .Append(context.Transforms.Concatenate("Features", "SymbolEncoded"))
                    .Append(context.Transforms.NormalizeMinMax("Features"))
                    .Append(context.Regression.Trainers.Sdca(
                        labelColumnName: "Price",
                        featureColumnName: "Features",
                        maximumNumberOfIterations: 100));

                var splitData = context.Data.TrainTestSplit(dataView, testFraction: 0.2, seed: 42);
                var model = pipeline.Fit(splitData.TrainSet);

                // Evaluate with try-catch
                try
                {
                    var predictions = model.Transform(splitData.TestSet);
                    var metrics = context.Regression.Evaluate(predictions, labelColumnName: "Price");

                    Console.WriteLine("Model Evaluation Metrics:");
                    Console.WriteLine($"R² Score: {metrics.RSquared:F4}");
                    Console.WriteLine($"Mean Absolute Error: {metrics.MeanAbsoluteError:F2}");
                    Console.WriteLine($"Root Mean Squared Error: {metrics.RootMeanSquaredError:F2}");

                    // Add back feature importance with correct column mapping
                    try
                    {
                        PrintFeatureImportance(context, model, splitData.TrainSet);
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"Warning: Feature importance calculation failed: {ex.Message}");
                    }

                    // Add back residuals analysis with correct column mapping
                    try
                    {
                        PlotResiduals(predictions);
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"Warning: Residuals analysis failed: {ex.Message}");
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Warning: Model evaluation failed: {ex.Message}");
                }

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

        private static CryptoPrediction Predict(ITransformer model, CryptoData input)
        {
            if (model == null)
                throw new ArgumentNullException(nameof(model));
            if (input == null)
                throw new ArgumentNullException(nameof(input));

            var context = new MLContext();
            var predictionEngine = context.Model.CreatePredictionEngine<CryptoData, CryptoPrediction>(model);
            return predictionEngine.Predict(input);
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

                // Create price chart with moving averages
                string outputPath = Path.Combine(OUTPUT_DIR, $"{symbol}_price_chart.png");
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

        private static List<ResidualData> GenerateResiduals(ITransformer model, List<CryptoData> data)
        {
            var context = new MLContext();
            var residuals = new List<ResidualData>();

            try
            {
                var dataView = context.Data.LoadFromEnumerable(data);
                var predictions = model.Transform(dataView);

                // Extract predictions
                var predictionColumn = predictions.Schema["Score"];
                var labelColumn = predictions.Schema["Price"];

                // Create a prediction cursor
                var cursor = predictions.GetRowCursor(new DataViewSchema.Column[] {
                    predictionColumn,
                    labelColumn
                });

                var labelGetter = cursor.GetGetter<float>(labelColumn);
                var predictionGetter = cursor.GetGetter<float>(predictionColumn);

                // Process each row
                while (cursor.MoveNext())
                {
                    float label = 0;
                    float prediction = 0;

                    labelGetter(ref label);
                    predictionGetter(ref prediction);

                    residuals.Add(new ResidualData
                    {
                        Price = label,
                        PredictedPrice = prediction
                    });
                }

                Console.WriteLine($"Generated {residuals.Count} residual data points for visualization");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error generating residuals: {ex.Message}");
            }

            return residuals;
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
}
// Add this at the end of your namespace, after the existing classes

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

// Remove the extra closing brace at the end of the file

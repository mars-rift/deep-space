using Microsoft.ML;
using Microsoft.ML.Data;
using CsvHelper;
using CsvHelper.Configuration;
using System.Globalization;
using System.Linq;
using System.Collections.Generic;
using System.IO;
using System;

namespace CryptoPredictor
{
    public class Program
    {
        private const string CSV_FILE_PATH = "crypto.csv";

        public static void Main(string[] args)
        {
            try
            {
                // Load and prepare data
                var cryptoData = LoadData();
                if (cryptoData?.Any() != true)
                {
                    Console.WriteLine("Error: No data found in the CSV file.");
                    return;
                }

                // Train the model
                var model = TrainModel(cryptoData);

                // Use the model for predictions (example)
                var prediction = Predict(model, new CryptoData { Symbol = "BTC", Price = 50000 });
                Console.WriteLine($"Predicted Best Price to Buy BTC: {prediction.PredictedPrice:C2}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"An unexpected error occurred: {ex.Message}");
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

using ScottPlot;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace CryptoPredictor
{
    public static class Visualization
    {
        public static void CreatePriceChart(List<CryptoData> data, string symbol, string outputPath = "price_chart.png")
        {
            if (data == null || !data.Any())
                throw new ArgumentException("No data provided for visualization");

            var filteredData = data.Where(d => d.Symbol == symbol).ToList();
            if (!filteredData.Any())
                throw new ArgumentException($"No data found for symbol {symbol}");

            var prices = filteredData.Select(d => d.Price).ToArray();
            var plot = new Plot(600, 400);

            plot.AddScatter(Enumerable.Range(0, prices.Length).Select(i => (double)i).ToArray(),
                            prices.Select(p => (double)p).ToArray());

            plot.Title($"{symbol} Price Trends");
            plot.XLabel("Time Period");
            plot.YLabel("Price ($)");
            plot.SaveFig(outputPath);

            Console.WriteLine($"Price chart created at: {Path.GetFullPath(outputPath)}");
        }

        public static void CreatePredictionVsActualChart(List<ResidualData> residuals, string outputPath = "prediction_vs_actual.png")
        {
            if (residuals == null || !residuals.Any())
                throw new ArgumentException("No residual data provided for visualization");

            var plot = new Plot(800, 600);

            // Scatter plot of predicted vs actual
            plot.AddScatter(
                residuals.Select(r => (double)r.Price).ToArray(),
                residuals.Select(r => (double)r.PredictedPrice).ToArray(),
                System.Drawing.Color.Blue
            );

            // Add ideal prediction line (y=x)
            double min = Math.Min(residuals.Min(r => r.Price), residuals.Min(r => r.PredictedPrice));
            double max = Math.Max(residuals.Max(r => r.Price), residuals.Max(r => r.PredictedPrice));
            plot.AddLine(min, min, max, max, System.Drawing.Color.Red, 1);

            plot.Title("Predicted vs Actual Prices");
            plot.XLabel("Actual Price");
            plot.YLabel("Predicted Price");
            plot.SaveFig(outputPath);

            Console.WriteLine($"Prediction vs actual chart created at: {Path.GetFullPath(outputPath)}");
        }
    }
}
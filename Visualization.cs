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

        public static void CreatePredictionVsActualChart(List<ResidualData> residuals, string symbol, string outputPath = "prediction_vs_actual.png")
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

            plot.Title($"{symbol} Predicted vs Actual Prices");
            plot.XLabel("Actual Price");
            plot.YLabel("Predicted Price");
            plot.SaveFig(outputPath);

            Console.WriteLine($"Prediction vs actual chart created at: {Path.GetFullPath(outputPath)}");
        }

        public static void CreateVolatilityChart(List<CryptoTimeSeriesData> data, string symbol, string outputPath = "volatility_chart.png")
        {
            if (data == null || !data.Any())
                throw new ArgumentException("No data provided for volatility visualization");

            var filtered = data.Where(d => d.Symbol == symbol).OrderBy(d => d.Timestamp).ToList();
            if (!filtered.Any())
                throw new ArgumentException($"No data for symbol {symbol}");

            var xs = filtered.Select(d => d.Timestamp.ToOADate()).ToArray();

            var v7 = filtered.Select(d => (double)(d.Volatility7 ?? 0f)).ToArray();
            var v14 = filtered.Select(d => (double)(d.Volatility14 ?? 0f)).ToArray();
            var v21 = filtered.Select(d => (double)(d.Volatility21 ?? 0f)).ToArray();

            var v7z = filtered.Select(d => (double)(d.Volatility7Z ?? 0f)).ToArray();
            var v14z = filtered.Select(d => (double)(d.Volatility14Z ?? 0f)).ToArray();
            var v21z = filtered.Select(d => (double)(d.Volatility21Z ?? 0f)).ToArray();

            var plt = new Plot(1000, 600);

            // Raw volatility lines
            plt.AddScatter(xs, v7, System.Drawing.Color.Blue, label: "Volatility7");
            plt.AddScatter(xs, v14, System.Drawing.Color.Green, label: "Volatility14");
            plt.AddScatter(xs, v21, System.Drawing.Color.Orange, label: "Volatility21");

            // Z-score lines as dashed
            var s1 = plt.AddScatter(xs, v7z, System.Drawing.Color.LightBlue, label: "Volatility7Z"); s1.LineStyle = ScottPlot.LineStyle.Dash;
            var s2 = plt.AddScatter(xs, v14z, System.Drawing.Color.LightGreen, label: "Volatility14Z"); s2.LineStyle = ScottPlot.LineStyle.Dash;
            var s3 = plt.AddScatter(xs, v21z, System.Drawing.Color.Khaki, label: "Volatility21Z"); s3.LineStyle = ScottPlot.LineStyle.Dash;

            plt.Title($"{symbol} Volatility (7/14/21) and Z-scores");
            plt.YLabel("Volatility");
            plt.XLabel("Date");
            plt.Legend();
            plt.SetAxisLimits(xMin: xs.FirstOrDefault(), xMax: xs.LastOrDefault());

            // Format X axis as dates if possible
            try { plt.XAxis.DateTimeFormat(true); } catch { }

            plt.SaveFig(outputPath);
            Console.WriteLine($"Volatility chart created at: {Path.GetFullPath(outputPath)}");
        }

        public static void CreateVolatilityIndexChart(List<(DateTime Timestamp, double Index)> series, string symbol, string outputPath = "volatility_index.png")
        {
            if (series == null || !series.Any())
                throw new ArgumentException("No data provided for volatility index visualization");

            var xs = series.Select(s => s.Timestamp.ToOADate()).ToArray();
            var ys = series.Select(s => s.Index).ToArray();

            var plt = new Plot(1000, 400);
            var scatter = plt.AddScatter(xs, ys, System.Drawing.Color.Purple, label: "Volatility Index");
            scatter.MarkerSize = 4;
            scatter.LineWidth = 2;

            // Add threshold lines at 33 and 66
            plt.AddHorizontalLine(33, System.Drawing.Color.DarkGray);
            plt.AddHorizontalLine(66, System.Drawing.Color.DarkGray);

            double minY = ys.Min();
            double maxY = ys.Max();
            double range = Math.Max(5, maxY - minY);
            double margin = range * 0.15;
            double yMin = Math.Max(0, minY - margin);
            double yMax = Math.Min(100, maxY + margin);
            if (yMax - yMin < 10)
            {
                yMax = Math.Min(100, yMin + 10);
            }

            plt.Title($"{symbol} Volatility Index");
            plt.XLabel("Date");
            plt.YLabel("Index (0-100)");
            plt.SetAxisLimits(yMin: yMin, yMax: yMax, xMin: xs.FirstOrDefault(), xMax: xs.LastOrDefault());
            try { plt.XAxis.DateTimeFormat(true); } catch { }
            plt.Legend();

            plt.SaveFig(outputPath);
            Console.WriteLine($"Volatility index chart created at: {Path.GetFullPath(outputPath)}");
        }
    }
}
# Deep-Space CryptoPredictor

A machine learning-based cryptocurrency prediction tool that utilizes real-time Bitcoin data to generate price insights and visualizations.

## 🚀 Features

- **Real-time Data**: Fetches current Ethereum pricing from CoinGecko and Binance APIs
- **Automatic Fallbacks**: Gracefully handles API failures by trying multiple sources
- **Advanced ML Modeling**: Uses ML.NET to create price prediction models
- **Technical Indicators**: Calculates moving averages, RSI, and other technical indicators
- **Data Visualization**: Generates charts for price trends and prediction accuracy
- **Data Preprocessing**: Cleans data and handles outliers automatically
- **Feature Engineering**: Creates derived features to improve model accuracy

## 🛠️ Technologies Used

- **C# / .NET**: Core application framework
- **ML.NET**: Microsoft's machine learning library for .NET
- **CoinGecko API**: Primary data source for Crypto prices
- **Binance API**: Secondary data source
- **CSVHelper**: For fallback data handling from files

## 📋 Prerequisites

- .NET 8.0 SDK or newer
- Visual Studio 2022 or VS Code with C# extension

## 🔧 Setup

1. Clone this repository
2. Open the solution in Visual Studio or VS Code
3. Restore NuGet packages
4. Build and run the application

```bash
https://github.com/mars-rift/deep-space.git
cd deep-space
dotnet restore
dotnet build
dotnet run
```

## 📊 How It Works

1. **Data Collection**: The application connects to cryptocurrency APIs to fetch Ethereum market data
   - First attempts to use CoinGecko (more reliable in most regions)
   - Falls back to Binance if needed
   - Can use local CSV files as a final fallback

2. **Data Preprocessing**:
   - Removes outliers and invalid values
   - Applies transformations (log, normalization)
   - Generates technical indicators (moving averages, etc.)

3. **Model Training**:
   - Trains an ML.NET regression model
   - Uses feature engineering to improve predictions
   - Implements cross-validation and evaluation metrics

4. **Visualization**:
   - Creates price charts with indicators
   - Displays prediction vs. actual comparison charts
   - Visualizes model residuals for error analysis

## 🔄 API Usage

### CoinGecko API
The application uses CoinGecko's free API to fetch OHLCV (Open-High-Low-Close-Volume) data:
```csharp
var url = $"{BaseUrl}/coins/ethereum/market_chart?vs_currency=usd&days={days}&interval=daily";
```

### Binance API
As a fallback, the application can use Binance's API:
```csharp
var url = $"{BaseUrl}/klines?symbol=ETHUSDT&interval={interval}&limit={limit}";
```

## 📈 Example Output

When run successfully, the application will:

1. Download the latest Ethereum price data
2. Generate technical indicators
3. Create a price chart in the output directory
4. Train a price prediction model
5. Display prediction accuracy metrics
6. Create a prediction vs. actual visualization
7. Provide a sample prediction for a given Ethereum price

## 🤝 Contributing

Contributions are welcome! Here's how you can help:
- Add support for additional cryptocurrencies
- Implement more technical indicators
- Improve model accuracy
- Add more visualization options

## 📄 License

MIT License

---

*Note: This project is for educational purposes only. Cryptocurrency investments carry high risk, and predictions should not be used as financial advice.*

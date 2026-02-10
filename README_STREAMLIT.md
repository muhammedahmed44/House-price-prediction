# Pakistan House Price Prediction - Streamlit Frontend

## Overview
This is a comprehensive Streamlit web application for predicting house prices in major Pakistani cities using machine learning. The application provides an intuitive interface for property price prediction with advanced data visualization and market insights.

## Features

### 🏠 Price Prediction
- **Interactive Input Form**: Easy-to-use form for entering property details
- **Multi-city Support**: Covers Islamabad, Lahore, Karachi, Rawalpindi, and Faisalabad
- **Property Types**: Supports Houses, Flats, Farm Houses, Upper/Lower Portions, and Penthouses
- **Smart Location Filtering**: Locations automatically filtered based on selected city
- **Real-time Predictions**: Instant price predictions with market comparisons

### 📊 Data Analysis Dashboard
- **Summary Statistics**: Key metrics about the property market
- **Property Distribution**: Visual breakdown by property type
- **City-wise Analysis**: Price comparisons across different cities
- **Area vs Price Correlation**: Scatter plots showing relationships

### 📈 Market Trends & Insights
- **Top Locations**: Most expensive areas ranked by average price
- **Property Type Analysis**: Detailed breakdown by property categories
- **Area Category Trends**: Price patterns across different area sizes
- **Market Intelligence**: Comprehensive market insights

### 🔍 Model Information
- **Performance Metrics**: Model accuracy and validation scores
- **Feature Details**: Information about analyzed property attributes
- **Training Data**: Details about the dataset used for training

## Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation Steps

1. **Clone or Download the Project**
   ```bash
   # Navigate to your project directory
   cd house-price-prediction-model-main
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure Required Files are Present**
   - `model.pkl` - Trained machine learning model
   - `cleaned_data.csv` - Cleaned training dataset
   - `app.py` - Streamlit application
   - `requirements.txt` - Python dependencies

4. **Run the Application**
   ```bash
   streamlit run app.py
   ```

5. **Access the Application**
   - The application will automatically open in your default web browser
   - If not, navigate to `http://localhost:8501`

## Usage Guide

### Making a Price Prediction

1. **Navigate to Price Prediction Tab**
   - Click on the "🏠 Price Prediction" tab

2. **Fill in Property Details**
   - **Property Type**: Select from dropdown (House, Flat, etc.)
   - **City**: Choose the city where property is located
   - **Location**: Select specific area/sector (filtered by city)
   - **Area Size**: Enter area in square feet
   - **Bedrooms**: Number of bedrooms
   - **Bathrooms**: Number of bathrooms
   - **Purpose**: For Sale or For Rent
   - **Area Category**: Select size category (0-5 Marla, 5-10 Marla, etc.)

3. **Get Prediction**
   - Click "🔮 Predict Price" button
   - View predicted price and market insights
   - Compare with similar properties

### Exploring Data Insights

1. **Data Analysis Tab**
   - View overall market statistics
   - Analyze property distributions
   - Compare prices across cities

2. **Market Trends Tab**
   - Discover top expensive locations
   - Understand property type patterns
   - Analyze area category trends

## Technical Architecture

### Frontend
- **Framework**: Streamlit
- **Styling**: Custom CSS with responsive design
- **Visualizations**: Matplotlib and Seaborn
- **Data Tables**: Pandas DataFrames

### Backend
- **Model**: Pre-trained machine learning model (model.pkl)
- **Data Processing**: Pandas and NumPy
- **Prediction Pipeline**: Scikit-learn compatible

### Data Features
The model analyzes the following property attributes:
- Property Type (House, Flat, Farm House, etc.)
- Location and City
- Geographic Coordinates (Latitude, Longitude)
- Area Size and Category
- Number of Bedrooms and Bathrooms
- Property Purpose (Sale/Rent)
- Date Information
- Price per Square Foot

## Model Performance
- **R² Score**: 0.92
- **RMSE**: 12.5%
- **MAE**: 8.3%
- **Accuracy**: 91.7%

## Dataset Information
- **Total Records**: 147,958+ property listings
- **Cities Covered**: 5 major Pakistani cities
- **Property Types**: 6 different categories
- **Data Source**: Zameen.com (Pakistan's top real estate platform)
- **Time Period**: Historical property listings

## Troubleshooting

### Common Issues

1. **Model Loading Error**
   - Ensure `model.pkl` file exists in the project directory
   - Check file permissions and path

2. **Data Loading Error**
   - Verify `cleaned_data.csv` is present
   - Check file format and encoding

3. **Port Already in Use**
   - Try running on a different port: `streamlit run app.py --server.port 8502`

4. **Missing Dependencies**
   - Reinstall requirements: `pip install -r requirements.txt --force-reinstall`

### Performance Optimization
- The application uses caching for model and data loading
- Predictions are optimized for real-time responses
- Visualizations are optimized for web display

## Future Enhancements
- **Map Integration**: Interactive maps for location selection
- **Price History**: Historical price trends for locations
- **Comparison Tool**: Side-by-side property comparisons
- **Mobile App**: Native mobile application
- **Advanced Filters**: More sophisticated search options
- **User Accounts**: Save predictions and favorites

## Support
For issues or questions regarding the application, please refer to the troubleshooting section or check the Streamlit documentation.

---

**Note**: This application is for educational and demonstration purposes. Property prices should be verified with real estate professionals before making any financial decisions.
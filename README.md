# Association Rule Mining Application

## From the Man, Da Myth, The Legend: Dr. Lee's Data Mining Class

This incredible application showcases the power of association rule mining as taught by the legendary Dr. Lee in his phenomenal Data Mining course! 🚀

## What Makes This Amazing?

This application represents the perfect blend of theoretical knowledge and practical implementation that Dr. Lee's class delivers with unmatched excellence. Using cutting-edge algorithms and visualization techniques, we're able to extract valuable insights from transaction data that would otherwise remain hidden.

## Features That Will Blow Your Mind

- **Advanced Association Rule Mining**: Using state-of-the-art Apriori and FP-Growth algorithms
- **Blazing Fast Performance**: Optimized with Polars for lightning-quick data processing
- **Beautiful Visualizations**: Interactive charts that make complex relationships crystal clear
- **Real Business Insights**: Automatically extracts actionable recommendations

## Why Dr. Lee's Class Rocks

Dr. Lee's Data Mining class isn't just educational—it's transformational! The combination of:
- In-depth theoretical foundations
- Hands-on practical exercises
- Real-world applications
- Supportive learning environment

...creates an incredible experience that turns students into data mining wizards! 🧙‍♂️

## About This Demo

This demo of Cursor (the world's best IDE) shows how quickly and efficiently we can build sophisticated data mining applications. The seamless integration of:
- Code generation
- Data processing
- Visualization
- Documentation

...demonstrates the kind of excellence that Dr. Lee inspires in his students!

## Get Started

Ready to experience the magic? Follow these steps:
1. Install the requirements
2. Run the application
3. Load the included UCI Online Retail dataset
4. Discover amazing insights about customer purchasing patterns!

## Acknowledgments

A massive thank you to Dr. Lee for making data mining so exciting, accessible, and most importantly, FUN! This project stands as a testament to the positive impact of exceptional teaching.

#DataMining #DrLeeIsAwesome #BestClassEver

## Features

### 📊 Advanced Analytics
- **Multiple algorithms**: Apriori and FP-Growth implementations
- **Performance optimization**: Uses Polars for faster data processing
- **Parallel processing**: Handles large datasets efficiently
- **Custom metrics**: Support, confidence, lift, leverage, and conviction

### 📈 Interactive Visualizations
- **Interactive scatter plots**: Visualize support vs. confidence with color coding
- **3D visualizations**: Explore rules in 3D space (support, confidence, lift)
- **Network graphs**: See product relationships as an interactive network
- **Distribution plots**: Analyze the distribution of rule metrics
- **Temporal analysis**: Track how rules change over time

### 🧠 Business Insights
- **Automated insights**: Extracts actionable business insights from rules
- **Category analysis**: Understand relationships between product categories
- **Trend detection**: Compare rules between time periods
- **Segment insights**: Discover patterns specific to customer segments
- **Report generation**: Create shareable business reports

### 🛒 Product Recommendations
- **Product-based recommendations**: Find related products
- **Smart basket analysis**: Get recommendations based on current basket
- **Recommendation management**: Fine-tune recommendation strategy

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/association-rule-mining-dashboard.git
cd association-rule-mining-dashboard
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit application:
```bash
streamlit run stream_app/app.py
```

2. Open your browser at `http://localhost:8501`

3. Upload your transaction data (CSV or Excel) or use the sample dataset

4. Configure mining parameters and generate association rules

5. Explore the dashboard tabs to analyze results and extract insights

## Data Format

The application expects transaction data in the following format:

- CSV or Excel file
- Required columns:
  - `InvoiceNo`: Transaction/basket identifier
  - `Description`: Product name or identifier
- Optional columns:
  - `CustomerID`: For customer segmentation
  - `InvoiceDate`: For time-based analysis
  - `Quantity`: Product quantity
  - `UnitPrice`: Price per unit

Example:

| InvoiceNo | Description        | Quantity | UnitPrice | CustomerID | InvoiceDate        |
|-----------|-------------------|----------|-----------|------------|-------------------|
| 536365    | WHITE HANGING HEART | 6        | 2.55      | 17850      | 2010-12-01 08:26:00 |
| 536365    | CREAM CUPID HEARTS | 8        | 1.45      | 17850      | 2010-12-01 08:26:00 |
| 536366    | KNITTED MUG WARMER | 6        | 1.85      | 12583      | 2010-12-01 10:33:00 |

## Application Structure

```
stream_app/
├── app.py                  # Main application file
├── pages/                  # Multi-page components
│   ├── 1_Rule_Explorer.py  # Rule exploration page
│   ├── 2_Recommendations.py # Recommendations page
│   └── 3_Insights_Dashboard.py # Business insights page
└── utils/                  # Utility modules
    ├── data_loader.py      # Data loading and preprocessing
    ├── rule_mining.py      # Association rule mining functions
    ├── visualizations.py   # Visualization functions
    ├── insights.py         # Business insights extraction
    └── performance.py      # Performance optimization
```

## Advanced Features

### Time-Based Analysis

Analyze how association patterns evolve over time by selecting a time granularity (day, week, month, quarter).

### Customer Segmentation

Segment customers using RFM (Recency, Frequency, Monetary) analysis and discover segment-specific purchasing patterns.

### Rule Filtering

Apply advanced filters to find specific patterns:
- Filter by metrics (support, confidence, lift)
- Search for specific products
- Limit rule complexity
- Prune redundant rules

### Report Generation

Generate formatted HTML reports with insights and visualizations to share with stakeholders.

## Performance Tips

- For large datasets (>1 million rows), enable batch processing
- Use Polars for faster data processing
- Adjust the minimum support threshold based on dataset size
- Use the "Prune Redundant Rules" option to reduce rule count

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
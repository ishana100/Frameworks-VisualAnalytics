import pandas as pd
import numpy as np
from pathlib import Path

def main():
    try:
        # 1. Read the data
        print("Reading data from sample_data.csv...")
        df = pd.read_csv("sample_data.csv")
        print("Original columns:", df.columns.tolist())
        
        # 2. Clean column names
        df.columns = [str(c).strip().lower() for c in df.columns]
        print("\nAfter cleaning column names:", df.columns.tolist())
        
        # 3. Check data types
        print("\nInitial data types:")
        print(df.dtypes)
        
        # 4. Convert date column
        print("\nSample date values before conversion:", df['date'].head().tolist())
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        # Debug: Print the first few date values before conversion
        print("\nFirst few date values before conversion:")
        print(df['date'].head().to_string())
        
        # Convert date column to datetime with error handling
        try:
            df['date'] = pd.to_datetime(df['date'], format='mixed', dayfirst=True, errors='coerce')
            
            # Check if any dates couldn't be converted
            null_dates = df['date'].isnull()
            if null_dates.any():
                print(f"\nWarning: Could not convert {null_dates.sum()} date(s). Sample of problematic rows:")
                print(df[null_dates].head())
                
                # Try to identify the problematic date format
                problem_dates = df.loc[null_dates, 'date'].head(1).values[0]
                print(f"\nExample of problematic date format: '{problem_dates}'")  
                
                # Try alternative date parsing
                print("\nTrying alternative date parsing...")
                df.loc[null_dates, 'date'] = pd.to_datetime(df.loc[null_dates, 'date'], errors='coerce')
                
                # Check again after second attempt
                still_null = df['date'].isnull().sum()
                if still_null > 0:
                    print(f"Still could not convert {still_null} dates. These rows will be dropped.")
        
        except Exception as e:
            print(f"Error during date conversion: {str(e)}")
            print("Trying with default date parsing...")
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        # Drop rows with invalid dates or missing values
        initial_count = len(df)
        df = df.dropna(subset=['date', 'amount', 'type'])
        if len(df) < initial_count:
            print(f"\nDropped {initial_count - len(df)} rows with missing or invalid data")
            
        # Debug: Print the first few date values after conversion
        print("\nFirst few date values after conversion:")
        print(df['date'].head().to_string())
        print("Date column type:", df['date'].dtype)
        
        # 6. Process the data
        df['type'] = df['type'].str.strip().str.title()
        df['signed_amount'] = np.where(
            df['type'].eq('Expense'), 
            -df['amount'].abs(),
            df['amount'].abs()
        )
        
        # 7. Verify date column is datetime before adding date features
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            print("\nError: Date column is not in datetime format. Attempting to fix...")
            try:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                if df['date'].isnull().all():
                    raise ValueError("Could not convert any dates to datetime format")
            except Exception as e:
                print(f"Failed to convert date column: {str(e)}")
                print("Please check your date format. Expected format: YYYY-MM-DD")
                print("Sample of date values:", df['date'].head().tolist())
                return
        
        # Now safely add date features
        try:
            print("\nAdding date features...")
            df['year'] = df['date'].dt.year
            df['month'] = df['date'].dt.month
            df['month_name'] = df['date'].dt.strftime('%B')
            df['day_of_week'] = df['date'].dt.day_name()
            
            # Sort by date for running balance
            df = df.sort_values('date')
            
            # Calculate running balance
            df['running_balance'] = df['signed_amount'].cumsum()
            
            # Generate monthly summary
            print("Generating monthly summary...")
            monthly_summary = (df.groupby(['year', 'month'], as_index=False)
                               .agg(
                                   transaction_count=('signed_amount', 'count'),
                                   total_income=('signed_amount', lambda s: s[s>0].sum() if not s.empty else 0),
                                   total_expense=('signed_amount', lambda s: -s[s<0].sum() if not s.empty and any(s<0) else 0),
                                   net_cashflow=('signed_amount', 'sum')
                               ))
            
            print("\nDate features added successfully!")
            print("Sample data with date features:")
            print(df[['date', 'year', 'month', 'month_name', 'day_of_week']].head())
            
        except Exception as e:
            print(f"\nError adding date features: {str(e)}")
            print("Date column values:", df['date'].head().tolist())
            print("Date column type:", df['date'].dtype)
            return
        
        # 8. Save results
        output_dir = Path('output')
        output_dir.mkdir(exist_ok=True)
        
        # Save transactions
        transactions_file = output_dir / 'transactions_clean.csv'
        df.to_csv(transactions_file, index=False, float_format='%.2f')
        print(f"\nSaved transactions to: {transactions_file}")
        
        # Save monthly summary
        monthly_file = output_dir / 'monthly_summary.csv'
        monthly_summary.to_csv(monthly_file, index=False, float_format='%.2f')
        print(f"Saved monthly summary to: {monthly_file}")
        
        # Print sample output
        print("\nSample of processed data:")
        print(df[['date', 'type', 'amount', 'signed_amount', 'running_balance']].head())
        
        print("\nMonthly Summary:")
        print(monthly_summary.to_string(index=False))
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("\nDebug info:")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
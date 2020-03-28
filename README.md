# Prepare

Input data from kaggle website: https://www.kaggle.com/c/airbnb-recruiting-new-user-bookings/data

# Run

```
% sw_vers
ProductName:	Mac OS X
ProductVersion:	10.15.3
BuildVersion:	19D76
% sysctl -n machdep.cpu.brand_string
Intel(R) Core(TM) i7-5557U CPU @ 3.10GHz

% python -V
Python 3.7.7
% virtualenv-3.7 env
% source env/bin/active

(env) % pip install -r requirements.txt
Requirement already satisfied: numpy in ./env/lib/python3.7/site-packages (from -r requirements.txt (line 1)) (1.18.2)
Requirement already satisfied: pandas in ./env/lib/python3.7/site-packages (from -r requirements.txt (line 2)) (1.0.3)
Requirement already satisfied: scikit-learn in ./env/lib/python3.7/site-packages (from -r requirements.txt (line 3)) (0.22.2.post1)
Requirement already satisfied: xgboost in ./env/lib/python3.7/site-packages (from -r requirements.txt (line 4)) (1.0.2)
Requirement already satisfied: pytz>=2017.2 in ./env/lib/python3.7/site-packages (from pandas->-r requirements.txt (line 2)) (2019.3)
Requirement already satisfied: python-dateutil>=2.6.1 in ./env/lib/python3.7/site-packages (from pandas->-r requirements.txt (line 2)) (2.8.1)
Requirement already satisfied: joblib>=0.11 in ./env/lib/python3.7/site-packages (from scikit-learn->-r requirements.txt (line 3)) (0.14.1)
Requirement already satisfied: scipy>=0.17.0 in ./env/lib/python3.7/site-packages (from scikit-learn->-r requirements.txt (line 3)) (1.4.1)
Requirement already satisfied: six>=1.5 in ./env/lib/python3.7/site-packages (from python-dateutil>=2.6.1->pandas->-r requirements.txt (line 2)) (1.14.0)

(env) % time python src/main.py
Index(['age', 'signup_flow', 'country_destination', 'affiliate_channel_api',
       'affiliate_channel_content', 'affiliate_channel_direct',
       'affiliate_channel_other', 'affiliate_channel_remarketing',
       'affiliate_channel_sem-brand', 'affiliate_channel_sem-non-brand',
       ...
       'signup_app_Android', 'signup_app_Moweb', 'signup_app_Web',
       'signup_app_iOS', 'signup_method_basic', 'signup_method_facebook',
       'signup_method_google', 'first_active_weekday',
       'date_account_created_weekday', 'date_first_booking_weekday'],
      dtype='object', length=134)
Start to train...
training done, time cost:  0:00:27.537433
predict done, time cost:  0:00:00.106128
score: 0.868
python src/main.py  35.28s user 1.11s system 99% cpu 36.645 total
```

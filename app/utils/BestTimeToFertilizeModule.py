import json as js
from time import sleep

import requests as rq


class BestTimeToFertilize:
    __BASE_URL = "https://api.weatherbit.io/v2.0/forecast/daily?"
    __API_KEY = "32ed013a4fe647559c0bd34e3f2a53df"
    
    
    def __init__(self, city_name = 'Pune', state_name = 'Maharashtra', days = 7):
        self.city_name = '+'.join(city_name.lower().strip().split())
        self.state_name = '+'.join(state_name.lower().strip().split())
        self.country_name = 'IN'
        self.days = days
        self.response = None
        self.response_code = None
        self.weather_data = list()
        
    def api_caller(self):
        try:
            complete_url = "{0}city={1}&state={2}&country={3}&key={4}&days={5}".format(self.__BASE_URL, self.city_name, self.state_name, self.country_name, self.__API_KEY, self.days)
            # print(complete_url)
            # while self.response == None:
            self.response = rq.get(complete_url)
            sleep(5)
            self.response_code = self.response.status_code
            return self.response_code
        except Exception as msg:
            print("api_caller():", msg)
            return -1
        
    
    def is_api_call_success(self):
        if self.response_code == 200:
            return True
        elif self.response_code == 204:
            print('Content Not available, error code: 204')
        return False
    

    def json_file_bulider(self):
        try:
            json_obj = self.response.json()
            with open('weather_data.json', 'w') as file:
                js.dump(json_obj, file, indent = 1, sort_keys = True)
            print("weather_data.json file build successfully")
        except Exception as msg:
            print("json_bulider():", msg)
            
    
    def best_time_fertilize(self):
        json_obj = self.response.json()
        
        # print("City:", json_obj['city_name'], "\n")

        prolonged_precip = 0
        prolonged_prob = 0
        heavy_rain_2d = False
        heavy_rain_chance_2d = 0
        precip_2d = 0
        precip_chance_2d = 0
        
        for i in range(self.days):
            date = json_obj['data'][i]['datetime']
            temp = json_obj['data'][i]['temp']
            max_temp = json_obj['data'][i]['max_temp']
            min_temp = json_obj['data'][i]['min_temp']
            high_temp = json_obj['data'][i]['high_temp']
            low_temp = json_obj['data'][i]['low_temp']
            rh = json_obj['data'][i]['rh']
            precip = json_obj['data'][i]['precip']
            prob = json_obj['data'][i]['pop']
            w_code = json_obj['data'][i]['weather']['code']
            w_desc = json_obj['data'][i]['weather']['description']
            i_code = json_obj['data'][i]['weather']['icon']
            # Additional weather parameters
            wind_spd = json_obj['data'][i]['wind_spd']
            wind_cdir_full = json_obj['data'][i]['wind_cdir_full']
            wind_dir = json_obj['data'][i]['wind_dir']# Wind speed (Default m/s)
            sunrise_ts = json_obj['data'][i]['sunrise_ts']  # Sunrise time unix timestamp (UTC)
            sunset_ts = json_obj['data'][i]['sunset_ts']  # Sunset time unix timestamp (UTC)
            vis = json_obj['data'][i]['vis']  # Visibility (default KM)
            slp = json_obj['data'][i]['slp']  # Average sea level pressure (mb)
            dewpt = json_obj['data'][i]['dewpt']  # Average dew point (default Celsius)

            prolonged_precip += precip
            prolonged_prob += prob

            count_2d = 0
            if i < 2:
                precip_2d += precip
                precip_chance_2d += prob
                if w_code in [202, 233, 502, 521, 522]:
                    heavy_rain_2d = True
                    heavy_rain_chance_2d += prob
                    count_2d += 1
                    heavy_rain_chance_2d //= count_2d
            
            di = {
                "Date": date,
                "Temperature": temp,
                "Maximum Temperature": max_temp,
                "Minimum Temperature ": min_temp,
                "High Temperature Day-time High": high_temp,
                "Low Temperature Night-time Low": low_temp,
                "Relative Humidity": rh,
                "Rainfall": precip,
                "Probability of Precipitation": prob,
                "Weather Description": w_desc,
                "Wind Direction (degrees)":wind_dir,
                "Wind Direction ": wind_cdir_full,
                "Wind Speed": wind_spd,
                "Sunrise Time (UTC)": sunrise_ts,
                "Sunset Time (UTC)": sunset_ts,
                "Visibility (KM)": vis,
                "Sea Level Pressure (mb)": slp,
                "Average Dew Point (Celsius)": dewpt
            }
            self.weather_data.append(di)

        prolonged_prob //= self.days
        precip_chance_2d //= 2

        

        if heavy_rain_2d:
            print("*"*21, "Warning !!!", "*"*21)
            print("Heavy Rain Chances within 2 days:", heavy_rain_chance_2d)
            print("Heavy Rainfall puts your fertilizer at risk.")
            print("*"*21, "Warning !!!", "*"*21)
            return ('Warning', 'Heavy Rain Alert', 'Heavy Rain Chances within two days from now is %d%%' % (heavy_rain_chance_2d))

        elif prolonged_precip > 12.7 and prolonged_prob >= 50:
            print("*"*21, "Warning !!!", "*"*21)
            print("Prolonged Rainfall of greater than 12.7 mm puts your fertilizer at risk.")
            print("*"*21, "Warning !!!", "*"*21)
            return ('Warning', 'Prolonged Rainfall Alert', 'Prolonged Rainfall of greater than 12.7 mm puts your fertilizer at risk. From now %.2f mm rainfall will receive for upcoming seven days, chances %d%%' % (prolonged_precip, prolonged_prob))
        else:
            print("-"*80)
            print("The amount of rain for 2 days, counting today:", precip_2d)
            print("Chances of rain for 2 days, counting today:", precip_chance_2d)
            print()
            return ('Message', 'Precipitation Amount', 'The amount of rain for 2 days, counting today is %.2f mm and chances is %d%%' % (precip_2d, precip_chance_2d))

        
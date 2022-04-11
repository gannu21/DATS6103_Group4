import scrapy
import json


class CarMaxSpider(scrapy.Spider):
    name = "carmax"
    start_urls = ['https://www.carmax.com/cars/all']

    headers = {
        'Accept': '*/*',
        'Accept-Encoding': 'gzip, deflate, br',
        'Accept-Language': 'en-US,en;q=0.5',
        'Connection':'keep-alive',
        'Content-Type': 'application/json',
        'Host': 'www.carmax.com',
        'Referer': 'https://www.carmax.com/cars/all',
        'Sec-Fetch-Dest': 'empty',
        'Sec-Fetch-Mode': 'cors',
        'Sec-Fetch-Site': 'same-origin',
        'TE': 'trailers',
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64; rv:99.0) Gecko/20100101 Firefox/99.0'
    }
    zipcodes = ['22191' ,'28227', '21162',  '21030' , '08052', '12205', '14623','98036', '04106', '55428', '89130', '68118', '87113','64055', '01923', '40515', '04106','50322']

    def parse(self, response):
        for zipcode in self.zipcodes:
            for step in range(0, 7000, 20):
                try:
                    url = f'https://www.carmax.com/cars/api/search/run?uri=/cars/all&skip={step}&take=20&zipCode={zipcode}&radius=radius-250&shipping=0&sort=best-match&scoringProfile=BestMatchScoreVariant3&visitorID=a3cb6bdc-30a8-450f-970e-fdaf1dba540a'
                    request = scrapy.Request(url, 
                        callback=self.parse_api,
                        headers=self.headers)
                    yield request
                except:
                    continue
    
    def parse_api(self, response):
        data = response.json()
        for car in data['items']:
            #  yield { 
            #      'Name': f"{car['year']} {car['make']} {car['model']}",
            #      'Year': car['year'],
            #      'Make': car['make'],
            #      'Price': car['basePrice'],
            #      'Mileage': car['mileage']
            #  }
            yield {
                'stock_Number'  :  car['stockNumber'],
                'vin'  :  car['vin'],
                'year'  :  car['year'],
                'make'  :  car['make'],
                'model'  :  car['model'],
                'body'  :  car['body'],
                'base_price'  :  car['basePrice'],
                'msrp'  :  car['msrp'],
                'mileage'  : car['mileage'],
                'storeName'  :  car['storeName'],
                'city' :  car['geoCity'],
                'state'  :  car['state'],
                'exteriorColor'  :  car['exteriorColor'],
                'interiorColor'  :  car['interiorColor'],
                'trasmission'  :  car['transmission'],
                'storezip'  :  car['storeZip'],
                'mpgCity'  :  car['mpgCity'],
                'mpgHighway'  :  car['mpgHighway'],
                'cylinders'  :  car['cylinders'],
                'engineType'  :  car['engineType'],
                'fuelType'  :  car['fuelType'],
                'horsepower'  :  car['horsepower'],
                'horsepowerRpm'  :  car['horsepowerRpm'],
                'engineSize'  :  car['engineSize'],
                'engineTorque'  :  car['engineTorque'],
                'engineTorqueRpm'  :  car['engineTorqueRpm'],
                'Type'  :  car['types'][0]
            }
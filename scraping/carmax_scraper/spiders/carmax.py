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
    zipcodes = ['36301',
                '36106',
                '36606',
                '35806',
                '35244',
                '85260',
                '85705',
                '85353',
                '85297',
                '85711',
                '92562',
                '92562',
                '93551',
                '93036',
                '94533',
                '94588',
                '94538',
                '95136',
                '95407',
                '94014',
                '94523',
                '92211',
                '91764',
                '91304',
                '91502',
                '90621',
                '91010',
                '95661',
                '95828',
                '95356',
                '92618',
                '92504',
                '93650',
                '90504',
                '93307',
                '92626',
                '92111',
                '92025',
                '90301',
                '92860',
                '95212',
                '80134',
                '80234',
                '80121',
                '80401',
                '80922',
                '80538',
                '19711',
                '33805',
                '33966',
                '34104',
                '32609',
                '32304',
                '32505',
                '34957',
                '34207',
                '33411',
                '33613',
                '32822',
                '33317',
                '33172',
                '33426',
                '32225',
                '33760',
                '33014',
                '33064',
                '32124',
                '32771',
                '32244',
                '32904',
                '34205',
                '34474',
                '34711',
                '30071',
                '31406',
                '31909',
                '30519',
                '31088',
                '30622',
                '30144',
                '30071',
                '30281',
                '30907',
                '30122',
                '30076',
                '30341',
                '83642',
                '62269',
                '61704',
                '60102',
                '60563',
                '60477',
                '60173',
                '60162',
                '62711',
                '60453',
                '60022',
                '46805',
                '46280',
                '46410',
                '50322',
                '66204',
                '67207',
                '40299',
                '40515',
                '70065',
                '71105',
                '70503',
                '70433',
                '70809',
                '20723',
                '21704',
                '21801',
                '20723',
                '21162',
                '20877',
                '20613',
                '21043',
                '20723',
                '21030',
                '49512',
                '55428',
                '55109',
                '38866',
                '39532',
                '39206',
                '63123',
                '63376',
                '64055',
                '65802',
                '68118',
                '89130',
                '89014',
                '89146',
                '89511',
                '08817',
                '87507',
                '87113',
                '14623',
                '12205',
                '14228',
                '11554',
                '28405',
                '28590',
                '27612',
                '28227',
                '28054',
                '27407',
                '28134',
                '28602',
                '27103',
                '28303',
                '27616',
                '28546',
                '44128',
                '45240',
                '43235',
                '45449',
                '43219',
                '74133',
                '73069',
                '73131',
                '97317',
                '97008',
                '97222',
                '17050',
                '18045',
                '19047',
                '17601',
                '19406',
                '29579',
                '29607',
                '29414',
                '29210',
                '37067',
                '38305',
                '37620',
                '38119',
                '37128',
                '37204',
                '37421',
                '38133',
                '37934',
                '37115',
                '79936',
                '77469',
                '78577',
                '78412',
                '75070',
                '76210',
                '75701',
                '76543',
                '79424',
                '75041',
                '77074',
                '76120',
                '77090',
                '75062',
                '77034',
                '78229',
                '78753',
                '76132',
                '78745',
                '77065',
                '75093',
                '78233',
                '77449',
                '76054',
                '84095',
                '22801',
                '24502',
                '24019',
                '23060',
                '23452',
                '20165',
                '22192',
                '23060',
                '23114',
                '22911',
                '22407',
                '23608',
                '99212',
                '98371',
                '98036',
                '98057',
                '98662',
                '53719',
                '53186',
                '53224',
                '53142'
                ]

    def parse(self, response):
        for zipcode in self.zipcodes:
            for step in range(0, 10000, 20):
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
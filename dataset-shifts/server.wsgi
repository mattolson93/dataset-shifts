#datashifts.wsgi
import sys 
sys.path.insert(0, '/var/www/html/datashifts')
sys.path.insert(0, '/var/www/html/datashifts/env/lib/python3.6/site-packages')



from server import *
application = create_app()

"""
Get Footprint Density Matrix
- Kljun et al. 2015
"""
# built-in modules
import os
import time
import copy
import logging
# 3-rd party modules
import numpy as np
import pandas as pd
from .FFP_Python.calc_footprint_FFP_climatology import FFP_climatology, get_contour_levels, get_contour_vertices
import fiona
from shapely import geometry
from pyproj import Transformer
import netCDF4


class structuredData:
    def __init__(self, **kwargs):
        for k, v in kwargs.items(): self.__dict__[k]=v
        pass

def transform_crs(*xy, crs_in="EPSG:4326", crs_out="EPSG:3035"):
    transformer = Transformer.from_crs(crs_in, crs_out)
    return transformer.transform(*xy)


def ffp_from_nc(path, dx=10, dy=None):
    if dy is None: dy=dx
    if isinstance(path, str):
        nc_dataset = netCDF4.Dataset(path)
    elif isinstance(path, netCDF4._netCDF4):
        nc_dataset = path
    else:
        return None

    loop_variable = {c: nc_dataset.variables[c][:].ravel() for c in nc_dataset.variables.keys()}
    shape_variable = {c: nc_dataset.variables[c].shape for c in nc_dataset.variables.keys()}

    FP = {"xr": loop_variable['x'], "yr": loop_variable['y'], 'fr': [], 'n': np.nan,
            "fclim_2d": loop_variable['footprint'].reshape(shape_variable['footprint'])}
    FP["x_2d"], FP["y_2d"] = np.meshgrid(loop_variable['x'], loop_variable['y'])
    
    FPi = {}
    for i, d in enumerate(loop_variable["timestep"]):
        FP_ = {k: v if k != 'fclim_2d' else v[i] for k, v in FP.items()}
        # Save contours
        FP_ = get_contour(FP_, dx, dy, np.linspace(0.10, 0.90, 9))
        FPi[d] = FP_
    
    return FPi, nc_dataset

def get_date_footprint(data=None, latlon=None, latlon_crs="EPSG:4326", shp=True, tif=True, read=True, save=True, 
                       direct_path=f'data/output/SAC/footprint/hh/', **kwargs):
    # importing the module
    import tracemalloc

    # starting the monitoring
    tracemalloc.start()
    
    logger = logging.getLogger('footprint.hh')

    tstart = time.time() 
    dx = kwargs.get('dx', 10) 
    dy = kwargs.get('dy', kwargs.get('dx', 10))
    domain = kwargs.get('domain', [-10000, 10000]*2)

    datasetToReturn = structuredData()
    datasetToReturn.data = {}
    
    if latlon is not None:
        transformer = Transformer.from_crs(latlon_crs, "EPSG:3035")
        datasetToReturn.data['y'], \
        datasetToReturn.data['x'] = transformer.transform(*latlon)
    else:
        datasetToReturn.data['y'], \
        datasetToReturn.data['x'] = (0, 0)

    #print('go for loop\t', np.round(time.time()-tstart, 3))
    last_d = '\n'
    info_t_startloop = time.time()
    for d in data.TIMESTAMP.to_list():
        info_t_startdate = time.time()
        info_m_startdate = tracemalloc.get_traced_memory()
        datasetToReturn.data[d] = {}

        if save and not os.path.exists(f'{direct_path}footprint_{d.strftime("%Y%m%dT%H%M")}.tif'):
            print('\x1B[2A\r', last_d, '\n', d, sep='', end='\n')
            assert data is not None, 'Please include data for footprint calculation.'
            site_data = data[(data.TIMESTAMP==d)]
            
            try:
                if ('zm' not in site_data.columns) and ('master_sonic_height' in site_data.columns): site_data['zm'] = site_data['master_sonic_height']
                if ('zd' not in site_data.columns) and ('displacement_height' in site_data.columns): site_data['zd'] = site_data['displacement_height']
                if ('z0' not in site_data.columns): site_data['z0'] = None
                #with suppress_stdout():
                site_FP = FFP_climatology(*[site_data[v].to_list() if isinstance(v, str) else [v]*len(site_data) if v else None for v in [
                    'zm', 'z0', 'U', 1000, 'MO_LENGTH', 'V_SIGMA', 'USTAR', 'WD']], domain=domain, dx=dx, rs=[i/10 for i in range (1, 10)], verbosity=0)
            except Exception as e:
                logger.info(f'{d} {str(e)}')
                last_d = ''
                continue
            site_FP = center_footprint(site_FP, (datasetToReturn.data['x'], datasetToReturn.data['y']))

            # Save tif
            if tif: footprint_to_tif(site_FP, 10, 10, f'{direct_path}footprint_{d.strftime("%Y%m%dT%H%M")}.tif')
            
            # displaying the memory
            logger.debug(str(d) + 'after tif' + str(tracemalloc.get_traced_memory()))
            
            # Save contours
            if shp: footprint_to_shp(site_FP, f'{direct_path}footprint_{d.strftime("%Y%m%dT%H%M")}.shp')
            
            # displaying the memory
            logger.debug(str(d) + 'after shp' + str(tracemalloc.get_traced_memory()))
            
            del site_FP, site_data
            # displaying the memory
            logger.debug(str(d) + 'after del' + str(tracemalloc.get_traced_memory()))
            
            info_m_afterdate = tracemalloc.get_traced_memory()
            last_d = f'{d} ({info_m_afterdate[0] - info_m_startdate[0]}, {info_m_afterdate[1] - info_m_startdate[1]})' + ' '*10
            
        if read:
            flag = 0
            if tif and os.path.exists(f'{direct_path}footprint_{d.strftime("%Y%m%dT%H%M")}.tif'):
                t0 = time.time()
                site_FP_tif = satelite.tifData()
                site_FP_tif = site_FP_tif.read(f'{direct_path}footprint_{d.strftime("%Y%m%dT%H%M")}.tif')
                datasetToReturn.data[d]['tif'] = site_FP_tif
                datasetToReturn.data[d]['fclim_2d'] = site_FP_tif.tif.read()[0]
                #logger.debug(f'Reading SHP took {np.round(time.time()-t0, 3)} s.')
            else:
                flag = 1
            
            if shp and os.path.exists(f'{direct_path}footprint_{d.strftime("%Y%m%dT%H%M")}.shp'):
                t0 = time.time()
                datasetToReturn.data[d]['rs'], \
                datasetToReturn.data[d]['yr'], \
                datasetToReturn.data[d]['xr'] = shp_to_footprint(f'{direct_path}footprint_{d.strftime("%Y%m%dT%H%M")}.shp')
                #logger.debug(f'Reading SHP took {np.round(time.time()-t0, 3)} s.')
            else:
                flag += 2
            
            if flag:
                logger.info(f'Footprint file: {direct_path}footprint_{d.strftime("%Y%m%dT%H%M")}.tif not found (SAVE mode: ' + ('on' if save else 'off') + ').')

    logger.info(f'Collecting all footprint took {np.round(time.time()-info_t_startloop, 3)} s.')

    # stopping the library
    tracemalloc.stop()

    return datasetToReturn.data

def get_site_footprint(data=None, latlon=None, latlon_crs="EPSG:4326", shp=True, tif=True, read=True, save=True,
                        direct_path='data/output/SAC/footprint/footprint_climatology.tif', **kwargs):
        # importing the module
    import tracemalloc

    # starting the monitoring
    tracemalloc.start()
    
    logger = logging.getLogger('footprint.cl')

    direct_path_tif = direct_path.rsplit('.', 1)[0] + '.tif'
    direct_path_shp = direct_path.rsplit('.', 1)[0] + '.shp'

    dx = kwargs.get('dx', 10) 
    dy = kwargs.get('dy', kwargs.get('dx', 10))
    domain = kwargs.get('domain', [-10000, 10000]*2)

    datasetToReturn = structuredData()
    datasetToReturn.data = {}
    
    if latlon is not None:
        transformer = Transformer.from_crs(latlon_crs, "EPSG:3035")
        datasetToReturn.data['y'], \
        datasetToReturn.data['x'] = transformer.transform(*latlon)
    else:
        datasetToReturn.data['y'], \
        datasetToReturn.data['x'] = (0, 0)
    
    info_t_startloop = time.time()
    if not os.path.exists(direct_path_tif):
        assert data is not None, 'Please include data for footprint calculation.'
        site_data = data.copy()
        if ('zm' not in site_data.columns) and ('master_sonic_height' in site_data.columns): site_data['zm'] = site_data['master_sonic_height']
        if ('zd' not in site_data.columns) and ('displacement_height' in site_data.columns): site_data['zd'] = site_data['displacement_height']
        if ('z0' not in site_data.columns): site_data['z0'] = None
        #with suppress_stdout():
        site_FP = FFP_climatology(*[site_data[v].to_list() if isinstance(v, str) else [v]*len(site_data) if v else None for v in [
            'zm', 'z0', 'U', 1000, 'MO_LENGTH', 'V_SIGMA', 'USTAR', 'WD']], domain=domain, dx=dx, rs=[i/10 for i in range (1, 10)], verbosity=0)
        
        site_FP = center_footprint(site_FP, (datasetToReturn.data['x'], datasetToReturn.data['y']))

        # Save tif
        if tif: footprint_to_tif(site_FP, 10, 10, direct_path_tif)
        
        # displaying the memory
        logger.debug('after tif' + str(tracemalloc.get_traced_memory()))
        
        # Save contours
        if shp: footprint_to_shp(site_FP, direct_path_shp)
        
        # displaying the memory
        logger.debug('after shp' + str(tracemalloc.get_traced_memory()))
        
        del site_FP, site_data
        # displaying the memory
        logger.debug('after del' + str(tracemalloc.get_traced_memory()))
        
        """
        site_FP_tif = satelite.tifData()
        site_FP_tif_transform = satelite.Affine(10, 0, np.min(site_FP['x_2d']), 0, -10, np.max(site_FP['y_2d']))
        site_FP_tif = site_FP_tif.tif_from_data(np.array([site_FP['fclim_2d']]), #.astype('uint8'),
                            'EPSG:3035', site_FP_tif_transform)#(np.min(site_FP['x_2d']), np.min(site_FP['y_2d'])))
        site_FP_tif.write(direct_path_tif, **site_FP_tif.tif.meta)

        # Save contours
        footprint_to_shp(site_FP, direct_path_shp)
        logger.info(f"New shapefile saved in: `{direct_path_shp}`.")
        """
    
    if read:
        site_FP_tif = satelite.tifData()
        site_FP_tif = site_FP_tif.read(direct_path_tif)
        datasetToReturn.data['tif'] = site_FP_tif
        datasetToReturn.data['fclim_2d'] = site_FP_tif.tif.read()[0]

        datasetToReturn.data['rs'], \
        datasetToReturn.data['yr'], \
        datasetToReturn.data['xr'] = shp_to_footprint(direct_path_shp)
    
    logger.info(f'Collecting all footprint took {np.round(time.time()-info_t_startloop, 3)} s.')

    # stopping the library
    tracemalloc.stop()

    return datasetToReturn.data


def agg(FPs, dx, dy, rs):
    if not isinstance(FPs, list):
        FPs = [FPs]
    FPs = [fpi for fpi in FPs if "fclim_2d" in fpi.keys()]
    if not FPs:
        return
    
    #dx = FPs[0].dx
    #dy = FPs[0].dy
    #rs = FPs[0].rs
    x_2d = np.array(FPs[0]['x_2d'])
    y_2d = np.array(FPs[0]['y_2d'])
    FP = FPs[0]

    n_valid = len(FPs)
    FP["fclim_2d"] = fclim_2d = dx*dy*np.nansum([fpi["fclim_2d"] / n_valid for fpi in FPs], axis=0)

    clevs = get_contour_levels(fclim_2d, dx, dy, rs)
    frs = [item[2] for item in clevs]
    xrs = []
    yrs = []
    for ix, fr in enumerate(frs):
        xr, yr = get_contour_vertices(x_2d, y_2d, fclim_2d, fr)
        if xr is None:
            frs[ix]  = None
        xrs.append(xr)
        yrs.append(yr)

    FP.update({"xr": xrs, "yr": yrs, 'fr': frs, 'n': n_valid})
    return FP

def rename_dataframe_for_kljun(data, datetime='TIMESTAMP', zm='zm', d='zd', z0='z0', 
                               u_mean='u_rot', L='L', sigma_v='v_var', u_star='u*', wind_dir='wind_dir'):
    return pd.DataFrame(
        {'yyyy': data[datetime].dt.year,
        'mm': data[datetime].dt.month,
        'day': data[datetime].dt.day,
        'HH_UTC': data[datetime].dt.hour,
        'MM': data[datetime].dt.minute,
        'zm': data[zm],
        'd': data[d],
        'z0': data[z0],
        'u_mean': data[u_mean],
        'L': data[L],
        'sigma_v': data[sigma_v],
        'u_star': data[u_star],
        'wind_dir': data[wind_dir]})

def get_contour(FP, dx, dy, rs):
    clevs = get_contour_levels(FP["fclim_2d"], dx, dy, rs)
    frs = [item[2] for item in clevs]
    xrs = []
    yrs = []
    for ix, fr in enumerate(frs):
        xr, yr = get_contour_vertices(FP["x_2d"], FP["y_2d"], FP["fclim_2d"], fr)
        if xr is None:
            frs[ix]  = None
        xrs.append(xr)
        yrs.append(yr)

    FP.update({"xr": xrs, "yr": yrs, 'fr': frs, 'rs': rs, 'n': FP["n"]})
    return FP
        
def center_footprint(FFPd, centre):
    FFPd = copy.deepcopy(FFPd)
    ## meters to coordinates
    cx = lambda x : None if x is None else list(map(cx,x)) if (isinstance(x, list) or len(x.shape)>1) else x + centre[0]
    cy = lambda x : None if x is None else list(map(cy,x)) if (isinstance(x, list) or len(x.shape)>1) else x + centre[1]
    
    for var in set(['xr', 'x_2d']) & set(FFPd.keys()):
        FFPd[var] = list(map(cx, FFPd[var]))
        
    for var in set(['yr', 'y_2d']) & set(FFPd.keys()):
        FFPd[var] = list(map(cy, FFPd[var]))
    
    for var in ['x_2d', 'y_2d']:
        FFPd[var] = np.array(FFPd[var])
    return FFPd

def footprint_to_tif(FFPd, dx, dy, dst_path, crs='EPSG:3035', **schema):
    import rasterio

    # Footprint into array (band, lon, lat)
    arr = np.array(FFPd['fclim_2d'])
    if len(arr.shape) < 3: arr = np.flip([arr], 1)

    # Write a .tif profile
    profile = {'driver': 'GTiff', 
                        'dtype': arr.dtype, 
                        'nodata': None, 
                        'width': arr.shape[2],
                        'height': arr.shape[1],
                        'count': arr.shape[0],
                        'crs': crs,
                        'transform': rasterio.Affine(dx, 0, np.min(FFPd['x_2d']), 0, -dy, np.max(FFPd['y_2d']))}
    profile.update({k: v for k, v in schema if k in profile.keys()})
    
    # Write into destin file
    with rasterio.open(dst_path, 'w+', **profile) as dst:
        # Write each array in product_dic to the rasterio dataset
        dst.write(arr)
    return

def footprint_to_shp(FFPd, path, crs='EPSG:3035', **schema):
    # TRANSFORM THIS INTO A FUNCTION
    # Define a polygon feature geometry with one attribute
    schema.update({
        'geometry': 'Polygon',
        'properties': {'rs': 'int'},
    })

    # Write a new Shapefile
    with fiona.open(path, 'w', 'ESRI Shapefile', schema, crs=crs) as c:
        ## If there are multiple geometries, put the "for" loop here
        order = {k: i for i, k in enumerate(FFPd['rs'])}
        for k in sorted(FFPd['rs'], reverse=True):
            if FFPd['xr'][order[k]] is not None:
                poly = geometry.Polygon(list(zip(FFPd['xr'][order[k]], FFPd['yr'][order[k]])))
                c.write({
                    'geometry': geometry.mapping(poly),
                    'properties': {'rs': int(k*100)},
                })
    return

def shp_to_footprint(path):
    FP_rs = []
    FP_xr = []
    FP_yr = []

    with fiona.open(path, 'r') as c:
        count = 0
        while True and count < 10**3:
            try:
                this_polygon = c.next()
            except StopIteration:
                break
            FP_rs.append(this_polygon['properties']['rs']/100)
            yr, xr = map(list, zip(*this_polygon['geometry']['coordinates'][0]))
            FP_xr.append(xr)
            FP_yr.append(yr)
            count += 1
    
    return FP_rs, FP_xr, FP_yr

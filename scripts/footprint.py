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
from .FFP_Python.calc_footprint_FFP_climatology import FFP_climatology, check_ffp_inputs, get_contour_levels, get_contour_vertices
import fiona
from shapely import geometry
from pyproj import Transformer
import xarray as xr
import rasterio 
from rasterio.io import MemoryFile
import datetime

DEFAULT_ATTRS = {
    None:{
        'Title': 'Flux Footprint', #f'Diurnal Footprints for {data_to_nc.sitename} at 30-min resolution',
        'Creation_Date': "", #datetime.datetime.now().strftime('%d-%b-%Y'),
        'Contact': 'Pedro Coimbra and Benjamin Loubet at ECOSYS, INRAE, AgroParisTech, Université Paris-Saclay, Palaiseau, France, pedro-henrique.herig-coimbra@inrae.fr and benjamin.loubet@inrae.fr',
        'Aknowledgement': 'This is the continuation of the work from Betty Molinier and Natascha Kljun, Centre for Environmental and Climate Science, Lund University, betty.molinier@cec.lu.se and natascha.kljun@cec.lu.se',
        'Conventions': 'CF-1.8',
        'Creator': 'Betty Molinier¹ (ORCID: 0000-0002-7212-4120), Natascha Kljun¹ (ORCID: 0000-0001-9650-2184), Pedro Coimbra² (ORCID: 0009-0008-6845-8735) and Benjamin Loubet² (ORCID: 0000-0001-8825-8775).\n1 Centre for Environmental and Climate Science, Lund University, Sweden.\n2 ECOSYS, INRAE, AgroParisTech, Université Paris-Saclay, Palaiseau, France',
        'Institution': 'Centre for Environmental and Climate Science, Lund University, Lund, Sweden\nECOSYS, INRAE, AgroParisTech, Université Paris-Saclay, Palaiseau, France',
        'Source': "", #'Ecosystem Thematic Centre (2024). ETC NRT Fluxes, Romainville, 2023-02-13–2024-04-30, ICOS Cities, https://hdl.handle.net/11676/ML3hTCCg5neiu2yw_HUF7AkW; Hersbach, H., et al. (2023): ERA5 hourly data on single levels from 1940 to present. Copernicus Climate Change Service (C3S) Climate Data Store (CDS), DOI: 10.24381/cds.adbb2d47 (Accessed on 05-May-2024)',
        'Model_Used': 'FFP, Kljun et al. (2015), doi:10.5194/gmd‐8‐3695‐2015',
        'Summary': "", #f'This file contains flux footprints for the {data_to_nc.sitename} flux tower in France at 30-minute temporal resolution. The name of the file includes the date of all footprints contained.',
        'Subjects': "", #'Flux footprints, atmospheric modelling, urban flux, ICOS Cities',
        'Coordinate_Reference_System': "", #'WGS 84',
        'crs_projection4': "", #'+proj=tmerc +lat_0=48.88514 +lon_0=2.42222 +k=1 +x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84 +units=m +no_defs',
        'crs_wkt': "", #'PROJCS["WGS_1984_Transverse_Mercator",GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["Degree",0.0174532925199433]],PROJECTION["Transverse_Mercator"],PARAMETER["latitude_of_origin",48.885140],PARAMETER["central_meridian",2.422220],PARAMETER["scale_factor",1],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["Meter",1]]',
        'Variables': 'Time, X, Y, Boundary Layer Height Quality Flag, Footprint Climatology',
        'Tower_Location_Latitude': np.nan,
        'Tower_Location_Longitude': np.nan,
        'Tower Height (m)': np.nan,
        'Frequency': '30 min'
    },
    'timestep': {'units': 'yymmddhhMM'},
    'x': {'long_name': 'x coordinate of projection',
                       'standard_name': 'projection_x_coordinate', 
                       'units': 'meters'},
    'y': {'long_name': 'y coordinate of projection',
                       'standard_name': 'projection_y_coordinate', 
                       'units': 'meters'},
    'footprint': {'long_name': "footprint",
                                "units": "per square meter"},
}

class structuredData:
    def __init__(self, **kwargs):
        for k, v in kwargs.items(): self.__dict__[k]=v
        pass

class fpData(structuredData):
    def __init__(self, **kwargs):
        for k, v in kwargs.items(): self.__dict__[k]=v
        self.data = None
        pass
    
    def read(self, m='data', *args, **kwargs):
        method = {'data': read_from_data, 'tif': read_from_tif, 'nc': read_from_nc, 'shp': read_from_shp}
        assert m in method.keys(), f'Method not found, please select between {", ".join(list(method.keys()))}.'
        self.data = method[m](*args, **kwargs)
        #self.FPnc = dict_to_nc(self.FP, {'footprint': {k: v for k, v in self.parameter.items() if k in ['dx', 'dy']}})
        return self
    
    def write(self, m='tif', *args, **kwargs):
        if self.data == None: print('Warning, data is empty!')
        method = {'tif': write_to_tif, 'shp': write_to_shp, 'nc': write_to_nc}
        assert m in method.keys(), f'Method not found, please select between {", ".join(list(method.keys()))}.'
        return method[m](self.data, *args, **kwargs)
    
    def from_data(self, *args, **kwargs):
        return read_from_data(*args, **kwargs)
    
    def from_tif(self, *args, **kwargs):
        return read_from_tif(*args, **kwargs)
    
    def from_nc(self, *args, **kwargs):
        return read_from_nc(*args, **kwargs)
    
    def to_tif(self, *args, **kwargs):
        return write_to_tif(*args, **kwargs)
    
    def to_shp(self, *args, **kwargs):
        return write_to_shp(*args, **kwargs)
    
    def to_nc(self, *args, **kwargs):
        return write_to_nc(*args, **kwargs)
    
    def agg(self, name='footprint_climatology'):
        dx = self.data['footprint'].attrs.get('dx', 1)
        dy = self.data['footprint'].attrs.get('dy', dx)
        footprints = np.array(list(self.data.footprint.to_numpy()))
        self.data[name] = (('x', 'y'), agg(footprints, dx, dy))
        return self
    
    def plot(self, ix=0):
        from matplotlib import pyplot as plt
        ix = max(min(len(self.data.footprint)-1, ix), 0)
        plt.imshow(self.data.footprint[ix])

class tifData(rasterio.DatasetReader):
    def __init__(self) -> None:
        pass
    
    def read(self, infile, memory=True):
        if memory:
            with rasterio.open(infile) as tif:
                memory_tif = MemoryFile().open(**tif.meta)
                memory_tif.write(tif.read())
                self.tif = memory_tif
        else:
            self.tif = rasterio.open(infile)
        return self
    
def transform_crs(*xy, crs_in="EPSG:4326", crs_out="EPSG:3035"):
    transformer = Transformer.from_crs(crs_in, crs_out)
    return transformer.transform(*xy)

def get_data_from_url(url=None, *args, **kwargs):    
    import requests
    import pandas as pd
    from zipfile import ZipFile
    from io import BytesIO

    # Send a GET request to the URL
    response = requests.get(url)
    
    # Check if the request was successful
    if response.status_code == 200:
        in_memory = BytesIO(response.content)
        with ZipFile(in_memory, 'r') as zf:
            data = pd.read_csv(BytesIO(zf.read(zf.filelist[0])), *args, **kwargs)
    else:
        print(f"Failed to download file: {response.status_code}")
    return data

def update_attrs_in_nc(data, attrs={}):
    attrs0 = copy.deepcopy(DEFAULT_ATTRS)
    data.attrs.update(attrs0.pop(None, {}))
    data.attrs.update(attrs.pop(None, {}))
    
    attrs1 = copy.deepcopy(DEFAULT_ATTRS)
    attrs1.update(attrs)
    for k, v in attrs1.items():
        if k in data.variables.keys():
            data[k].attrs.update(v)
    return data


def fp_to_nc(fclim_2d, x, y, timestep, attrs={}):
    data = xr.Dataset({'footprint': (('timestep','x', 'y'), np.array(fclim_2d))},
                      coords={'timestep': np.array(timestep), 'x': np.array(x), 'y': np.array(y)})
    data = update_attrs_in_nc(data, attrs)
    return data

def dict_to_nc(FFPd, attrs={}):
    if FFPd == {}: return None
    x = np.array(list(FFPd[list(FFPd.keys())[0]].get('x_2d')))[0, :]
    y = np.array(list(FFPd[list(FFPd.keys())[0]].get('y_2d')))[:, 0]
    footprints = [f['fclim_2d'] for f in FFPd.values()]
    timesteps = list(FFPd.keys())
    data = fp_to_nc(footprints, x, y, timesteps, attrs=attrs)
    return data

def nc_to_dict(nc_dataset, dx=None, dy=None):
    if dx is None: dx = nc_dataset['footprint'].attrs.get('dx', 10)
    if dy is None: dy = nc_dataset['footprint'].attrs.get('dy', dx)
    FPi = {}

    loop_variable = {c: nc_dataset.variables[c].to_numpy() for c in nc_dataset.variables.keys()}

    FP = {"xr": loop_variable['x'], "yr": loop_variable['y'], 'fr': [], 'n': np.nan,
            "fclim_2d": loop_variable['footprint']}
    FP["x_2d"], FP["y_2d"] = np.meshgrid(loop_variable['x'], loop_variable['y'])
    
    for i, d in enumerate(loop_variable["timestep"]):
        FP_ = {k: v if k != 'fclim_2d' else v[i] for k, v in FP.items()}
        # Save contours
        FP_ = get_contour(FP_, dx, dy, np.linspace(0.10, 0.90, 9))
        FPi[d] = FP_

    #loc = (nc_dataset.Tower_Location_Latitude, nc_dataset.Tower_Location_Longitude)
    #loc0 = transform_crs(loc[0], loc[1], crs_in=nc_dataset.crs_wkt)
    
    attr = {None: nc_dataset.attrs}
    attr.update({c: nc_dataset[c].attrs for c in nc_dataset.variables.keys()})
    return FPi, attr#{'dx': dx, 'dy': dy, 'loc': loc, 'loc0': loc0, 'crs': rasterio.crs.CRS.from_wkt(nc_dataset.crs_wkt)}

def read_from_data(data=None, by=None, latlon=None, latlon_crs="EPSG:4326", dst_crs="EPSG:3035", **kwargs):    
    logger = logging.getLogger('footprint.from_data')

    required_columns = ['zm', 'z0', 'WS', 'PBLH', 'MO_LENGTH', 'V_SIGMA', 'USTAR', 'WD']
    for c in required_columns:
        if c not in data.dropna(axis=1, how='all').columns: 
            logger.debug(f'Columns {c} not found in data.')
            data[c] = kwargs.get(c, np.nan)

    dx = kwargs.get('dx', 10) 
    dy = kwargs.get('dy', dx)
    domain = kwargs.get('domain', [-500, 500]*2)
    
    if latlon is not None:
        transformer = Transformer.from_crs(latlon_crs, dst_crs)
        laty, lonx = transformer.transform(*latlon)
    else:
        laty, lonx = (0, 0)
    
    group_data = [('climatology', data)] if by == None else data.groupby(by)
    
    FPi = {}
    for i, dati in group_data:
        assert dati is not None, 'Please include data for footprint calculation.'
        
        try:
            FP = FFP_climatology(*[dati[v].to_list() if isinstance(v, str) else [v]*len(dati) if v else None 
                                   for v in required_columns], 
                                   domain=domain, dx=dx, dy=dy, rs=[i/10 for i in range (1, 10)], verbosity=0)
        except Exception as e:
            logger.info(f'{i} {str(e)}')
            continue
        FP = center_footprint(FP, (lonx, laty))
        FPi[i] = FP
    
    fp_crs = rasterio.crs.CRS.from_string(dst_crs)
    data = dict_to_nc(FPi, {'footprint': {'dx, dy': (dx, dy), 'tower': latlon, 'tower_xy': (laty, lonx)},
                       None: {'Coordinate_Reference_System': fp_crs.to_string(),
                              'crs_projection4': fp_crs.to_proj4(),
                              'crs_wkt': fp_crs.to_wkt()}})
    return data
    #return FPi, data, {'dx': dx, 'dy': dy, 'loc': latlon, 'loc0': (laty, lonx), 'crs': rasterio.crs.CRS.from_string(dst_crs)}

def read_from_nc(path, dx=10, dy=None):
    if dy is None: dy=dx
    FPi = {}

    if isinstance(path, str):
        nc_dataset = xr.open_dataset(path, engine="netcdf4")
    elif isinstance(path, (xr.core.dataset.Dataset)):
        nc_dataset = path
    else:
        return None
    return nc_dataset

    loop_variable = {c: nc_dataset[c].to_numpy() for c in nc_dataset.variables.keys()}

    FP = {"xr": loop_variable['x'], "yr": loop_variable['y'], 'fr': [], 'n': np.nan,
            "fclim_2d": loop_variable['footprint']}
    FP["x_2d"], FP["y_2d"] = np.meshgrid(loop_variable['x'], loop_variable['y'])
    
    for i, d in enumerate(loop_variable["timestep"]):
        FP_ = {k: v if k != 'fclim_2d' else v[i] for k, v in FP.items()}
        # Save contours
        FP_ = get_contour(FP_, dx, dy, np.linspace(0.10, 0.90, 9))
        FPi[d] = FP_

    loc = (nc_dataset.Tower_Location_Latitude, nc_dataset.Tower_Location_Longitude)
    loc0 = transform_crs(loc[0], loc[1], crs_in=nc_dataset.crs_wkt)
    return FPi, nc_dataset, {'dx': dx, 'dy': dy, 'loc': loc, 'loc0': loc0, 'crs': rasterio.crs.CRS.from_wkt(nc_dataset.crs_wkt)}

def read_from_tif(path, latlon=None, latlon_crs="EPSG:4326", **kwargs):
    assert os.path.exists(path), 'Path does not exist.'
    
    dx = kwargs.get('dx', 10) 
    dy = kwargs.get('dy', dx)
    rs = kwargs.get('rs', [i/10 for i in range(1, 9)])

    group_path = {p: path + p for p in os.listdir(path) if p.endswith('.tif')} if os.path.isdir(path) else {os.path.basename(path): path}

    FPi = {}
    tif = {}
    for i, pati in group_path.items():
        site_FP_tif = tifData().read(pati).tif
        tif[i] = site_FP_tif
        FPi[i] = {'fclim_2d': site_FP_tif.read()[0]}
        #logger.debug(f'Reading SHP took {np.round(time.time()-t0, 3)} s.')
        x = np.linspace(tif[i].bounds[0], tif[i].bounds[2], FPi[i]['fclim_2d'].shape[0])
        y = np.linspace(tif[i].bounds[1], tif[i].bounds[3], FPi[i]['fclim_2d'].shape[1])
        FPi[i]["x_2d"], FPi[i]["y_2d"] = np.meshgrid(x, y)
        #FPi[i] = get_contour(FPi[i], dx, dy, rs)
        
    fp_crs = site_FP_tif.crs
    if latlon is not None:
        transformer = Transformer.from_crs(latlon_crs, fp_crs)
        laty, lonx = transformer.transform(*latlon)
    else:
        laty, lonx = (0, 0)
        
    for i, f in FPi.items(): FPi[i] = center_footprint(f, (lonx, laty))

    data = dict_to_nc(FPi, {'footprint': {'dx, dy': (dx, dy), 'tower': latlon, 'tower_xy': (laty, lonx)},
                       None: {'Coordinate_Reference_System': fp_crs.to_string(),
                              'crs_projection4': fp_crs.to_proj4(),
                              'crs_wkt': fp_crs.to_wkt()}})
    return data
    #return FPi, tif, {'dx': dx, 'dy': dy, 'loc': (None, None), 'loc0': (None, None), 'crs': site_FP_tif.tif.crs}


def read_from_shp(path, name='climatology', bounds=(), shape=()):
    from scipy.interpolate import griddata
    FP_rs = []
    FP_xr = []
    FP_yr = []
    FP_polygon = []
    FP_area = []
    data = []

    with fiona.open(path, 'r') as c:
        for count, feature in enumerate(c):
            if count > 10**3:
                print('break')
                break
            fr = feature['properties']['fr']
            FP_rs.append(feature['properties']['rs'] / 100)
            yr, xr = map(list, zip(*feature['geometry']['coordinates'][0]))
            FP_xr.append(xr)
            FP_yr.append(yr)
            FP_polygon.append(feature)
            FP_area.append(0.5*np.abs(np.dot(xr,np.roll(yr,1))-np.dot(yr,np.roll(xr,1))))
            data.append(tuple(zip(xr, yr, [fr]*len(xr))))
    
    FP_area = [l - FP_area[i+1] if i < len(FP_area)-1 else l for i, l in enumerate(FP_area)]

    # flatten data
    data = [d for el in data for d in el]
    points = np.array([(x, y) for x, y, _ in data])
    values = np.array([z for _, _, z in data])

    # Define the grid dimensions (adjust as needed)
    grid_y, grid_x = np.mgrid[bounds[0]:bounds[1]:np.complex64(shape[0]), 
                              bounds[2]:bounds[3]:np.complex64(shape[1])]  # 10x10 grid from (-10000,-10000) to (10000,10000)
    grid_z = griddata(points, values, (grid_y, grid_x), method='cubic')

    data = dict_to_nc({name: {"xr": FP_xr, "yr": FP_yr, "fr": FP_rs, 'geom': FP_polygon,
                       "x_2d": grid_x, "y_2d": grid_y, "fclim_2d": grid_z}})
    return data

def write_to_tif(FFPd, dx, dy, dst_path, crs='EPSG:3035', **schema):
    import rasterio

    if isinstance(FFPd, dict): FFPd = dict_to_nc(FFPd)
    #    arr = np.array(FFPd['fclim_2d'])
    #    x = np.array(FFPd['x_2d'])[0, :]
    #    y = np.array(FFPd['y_2d'])[:, 0]
    #elif isinstance(FFPd, xr.core.dataset.Dataset):
    arr = FFPd['footprint'].to_numpy()
    x = FFPd['x'].to_numpy()
    y = FFPd['y'].to_numpy()

    # Footprint into array (band, lon, lat)
    #arr = np.array(FFPd['fclim_2d'])
    if len(arr.shape) < 3: arr = [arr]
    arr = np.flip(arr, 1)

    # Write a .tif profile
    profile = {'driver': 'GTiff', 
                        'dtype': arr.dtype, 
                        'nodata': None, 
                        'width': arr.shape[2],
                        'height': arr.shape[1],
                        'count': arr.shape[0],
                        'crs': crs,
                        'transform': rasterio.Affine(dx, 0, np.min(x), 0, -dy, np.max(y))}
    profile.update({k: v for k, v in schema if k in profile.keys()})
    
    # Write into destin file
    with rasterio.open(dst_path, 'w+', **profile) as dst:
        # Write each array in product_dic to the rasterio dataset
        dst.write(arr)
    return

def write_to_shp(FFPd, path, crs='EPSG:3035', **schema):
    if isinstance(FFPd, xr.core.dataset.Dataset):
        FFPd, _ = nc_to_dict(FFPd)
    
    # Define a polygon feature geometry with one attribute
    schema.update({
        'geometry': 'Polygon',
        'properties': {'rs': 'int', 'fr': 'float'},
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
                    'properties': {'rs': int(k*100), 'fr': FFPd['fr'][order[k]]},
                })
    return

def write_to_nc(FFPd, path, attrs={}):
    if isinstance(FFPd, dict): FFPd = dict_to_nc(FFPd)
    FFPd = update_attrs_in_nc(FFPd, attrs)
    FFPd.to_netcdf(path, 'w')
    return

def from_data_to_tif(url='https://data.icos-cp.eu/licence_accept?ids=%5B%22mAl0uywRaIcP-maLY7cJ3Q_U%22%5D', 
                     dst_path='sample/output/from_data_portal/FR-Gri/FR-Gri_footprint_{}.tif', 
                     zm=4, latlon=(48.84422, 1.95191), **kwargs):
    data_to_tif = fpData()
    data_to_tif.data = get_data_from_url(url, na_values=[-9999]).dropna(subset=['WS', 'WD'])

    for d, data in data_to_tif.data.groupby('TIMESTAMP_END'):
        print(d, '<'+'-'*15, ' '*10, end='\r')

        if os.path.exists(dst_path.format(f'{d}')): continue
        this_nc = read_from_data(
            data, by='TIMESTAMP_END', **{'zm': zm, 'z0': None, 'PBLH': 1000, 'latlon': latlon}, **kwargs)
        if this_nc is None: continue
        dx, dy = this_nc.footprint.attrs['dx, dy']
        write_to_tif(this_nc, dx, dy, dst_path.format(f'{d}'))
        del this_nc
    return data_to_tif


def agg(fclim_2d, dx, dy):
    fclim_2d = np.array(fclim_2d)
    if len(fclim_2d.shape) == 2:
        print(f"Footprint must be 3D (time, x, y), dimension passed was: {fclim_2d.shape}.")
        return fclim_2d
    
    assert len(fclim_2d.shape) == 3, f"Footprint must be 3D (time, x, y), dimension passed was: {fclim_2d.shape}."
    n_valid = len(fclim_2d)
    fclim_clim = dx*dy*np.nansum([fpi / n_valid for fpi in fclim_2d], axis=0)
    return fclim_clim

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

    FP.update({"xr": xrs, "yr": yrs, 'fr': frs, 'rs': rs})
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

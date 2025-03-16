Variables stored in separate files (Header+values)

Filename

	Data_separate_files_header_startdate(YYYYMMDD)_enddate(YYYYMMDD)_userid_randomstring_currrentdate(YYYYMMDD).zip
	
	e.g., Data_separate_files_header_20050316_20050601.zip

	
Folder structure

	Networkname
		Stationname

		
Dataset Filename

	CSE_Network_Station_Variablename_depthfrom_depthto_startdate_enddate.ext

	CSE	- Continental Scale Experiment (CSE) acronym, if not applicable use Networkname
	Network	- Network abbreviation (e.g., OZNET)
	Station	- Station name (e.g., Widgiewa)
	Variablename - Name of the variable in the file (e.g., Soil-Moisture)
	depthfrom - Depth in the ground in which the variable was observed (upper boundary)
	depthto	- Depth in the ground in which the variable was observed (lower boundary)
	startdate -	Date of the first dataset in the file (format YYYYMMDD)
	enddate	- Date of the last dataset in the file (format YYYYMMDD)
	ext	- Extension .stm (Soil Temperature and Soil Moisture Data Set see CEOP standard)
	
	e.g., OZNET_OZNET_Widgiewa_Soil-Temperature_0.150000_0.150000_20010103_20090812.stm

	
File Content Sample
	
	REMEDHUS   REMEDHUS        Zamarron          41.24100    -5.54300  855.00    0.05    0.05  (Header)
	2005/03/16 00:00    10.30 U	M	(Records)
	2005/03/16 01:00     9.80 U M

	
Header

	CSE Identifier - Continental Scale Experiment (CSE) acronym, if not applicable use Networkname
	Network	- Network abbreviation (e.g., OZNET)
	Station	- Station name (e.g., Widgiewa)
	Latitude - Decimal degrees. South is negative.
	Longitude - Decimal degrees. West is negative.
	Elevation - Meters above sea level
	Depth from - Depth in the ground in which the variable was observed (upper boundary)
	Depth to - Depth in the ground in which the variable was observed (lower boundary)

	
Record

	UTC Actual Date and Time
	yyyy/mm/dd HH:MM
	Variable Value
	ISMN Quality Flag
	Data Provider Quality Flag, if existing


Network Information

	AACES
		Abstract: 
		Continent: Oceania
		Country: Australia
		Stations: 49
		Status: inactive
		Data Range: from 2010-01-18  to 2010-09-26
		Type: project
		Url: http://www.moisturemap.monash.edu.au/
		Reference: Peischl, S., Walker, J. P., Rüdiger, C., Ye, N., Kerr, Y. H., Kim, E., Bandara, R., and Allahmoradi, M.: The AACES field experiments: SMOS calibration and validation across the Murrumbidgee River catchment, Hydrology and Earth System Sciences, Discuss., 9, 2763-2795, https://doi.org/10.5194/hess-16-1697-2012, 2012;
		Variables: precipitation, soil temperature, soil moisture, 
		Soil Moisture Depths: 0.00 - 0.05 m, 0.00 - 0.06 m, 0.25 - 0.25 m
		Soil Moisture Sensors: ThetaProbe ML2X, 

	AMMA-CATCH
		Abstract: This network consists of three supersites in Benin, Niger and Mali. Mali works operational since 2005, Niger and Benin since 2006. Several measurements in Mali and Niger are taken at the same station with the same sensor type in the same depth. They are located at the bottom (sensor CS616_1), middle (CS616_2) or top (CS616_3) of dune slopes (Mali) or from a plateau to the valley bottom (Niger).
		Continent: Africa
		Country: Benin, Niger, Mali
		Stations: 7
		Status: running
		Data Range: from 2005-01-01 
		Type: project
		Url: http://www.amma-catch.org
		Reference: Mougin, E., Hiernaux, P., Kergoat, L., Grippa, M., de Rosnay, P., Timouk, F., Le Dantec, V., Demarez, V., Lavenu, F., Arjounin, M., Lebel, T. et al., 2009. The AMMA-CATCH Gourma observatory site in Mali: Relating climatic variations to changes in vegetation, surface in press, hydrology, fluxes and natural resources. Journal of Hydrology, 375(1-2): 14-33, https://doi.org/10.1016/j.jhydrol.2009.06.045;

Cappelaere, C., Descroix, L., Lebel, T., Boulain, N., Ramier, D., Laurent, J.-P., Le Breton, E., Boubkraoui, S., Bouzou Moussa, I. et al., 2009. The AMMA Catch observing system in the cultivated Sahel of South West Niger- Strategy, Implementation and Site conditions, 2009. Journal of Hydrology, 375(1-2): 34-51, https://doi.org/10.1016/j.jhydrol.2009.06.021;

de Rosnay, P., Gruhier, C., Timouk, F., Baup, F., Mougin, E., Hiernaux, P., Kergoat, L., and LeDantec, V.: Multi-scale soil moisture measurements at the Gourma meso-scale site in Mali, Journal of Hydrology, 375, 241-252, 2009, https://doi.org/10.1016/j.jhydrol.2009.01.015;

Lebel, Thierry, Cappelaere, Bernard, Galle, Sylvie, Hanan, Niall, Kergoat, Laurent, Levis, Samuel, Vieux, Baxter, Descroix, Luc, Gosset, Marielle, Mougin, Eric, Peugeot, Christophe and Seguis, Luc, 2009: AMMA-CATCH studies in the Sahelian region of West-Africa: An overview. JOURNAL OF HYDROLOGY, 375, 3-13, https://doi.org/10.1016/j.jhydrol.2009.03.020;

Galle, S., Grippa, M., Peugeot, C., Bouzou Moussa, I., Cappelaere, B., Demarty, J., Mougin, E., Lebel, T. and Chaffard, V., 2015, December. AMMA-CATCH a Hydrological, Meteorological and Ecological Long Term Observatory on West Africa: Some Recent Results. In AGU Fall Meeting Abstracts (Vol. 2015, pp. GC42A-01).;
		Variables: soil moisture, 
		Soil Moisture Depths: 0.05 - 0.05 m, 0.10 - 0.10 m, 0.10 - 0.40 m, 0.20 - 0.20 m, 0.40 - 0.40 m, 0.40 - 0.70 m, 0.60 - 0.60 m, 0.70 - 1.00 m, 1.00 - 1.00 m, 1.00 - 1.30 m, 1.05 - 1.35 m, 1.20 - 1.20 m
		Soil Moisture Sensors: CS616, 

	ARM
		Abstract: The soil moisture datasets collected at ARM facilities originates from two different instruments. SWATS instrument measure soil moisture in two different profiles at 8 depths from 0,05 to 1,75m, the SEBS instrument measure three profiles in depth of 0,025m. The site is managed by U.S. Department of Energy as part of the Atmospheric Radiation Measurement Climate Research Facility.
		Continent: Americas
		Country: USA
		Stations: 35
		Status: running
		Data Range: from 1996-02-05 
		Type: project
		Url: http://www.arm.gov/
		Reference: Cook, D. R. (2016a), Soil temperature and moisture proﬁle (stamp) system handbook, Technical report, DOE Oﬃce of Science Atmospheric Radiation Measurement (ARM) Program. https://www.osti.gov/biblio/1332724;

Cook, D. R. (2016b), Soil water and temperature system (swats) instrument handbook, Technical report, DOE Oﬃce of Science Atmospheric Radiation Measurement (ARM) Program. https://www.osti.gov/biblio/1004944;

Cook, D. R. & Sullivan, R. C. (2018), Surface energy balance system (sebs) instrument handbook, Technical report, DOE Office of Science Atmospheric Radiation Measurement (ARM) Program. https://www.arm.gov/publications/tech_reports/handbooks/sebs_handbook.pdf;
		Variables: precipitation, soil temperature, air temperature, soil moisture, 
		Soil Moisture Depths: 0.02 - 0.02 m, 0.05 - 0.05 m, 0.10 - 0.10 m, 0.15 - 0.15 m, 0.20 - 0.20 m, 0.25 - 0.25 m, 0.35 - 0.35 m, 0.50 - 0.50 m, 0.60 - 0.60 m, 0.75 - 0.75 m, 0.80 - 0.80 m, 0.85 - 0.85 m, 1.25 - 1.25 m, 1.75 - 1.75 m
		Soil Moisture Sensors: Hydraprobe II Sdi-12 E, Hydraprobe II Sdi-12 S, Hydraprobe II Sdi-12 W, SMP1, Water Matric Potential Sensor 229L, 

	AWDN
		Abstract: In 1998 the High Plains Regional Climate Center upgraded stations in the Automated Weather Data Network (AWDN) to monitor soil water at 14 sites in Nebraska. The Natural Resources Conservation Service (NRCS) has provided soil moisture sensors for some of the sites. Currently, sensors are placed at 10, 25, 50, and 100 cm below the surface at each site. AWDN technicians maintain the sensors and laboratory facilities were updated to include adequate test and calibration equipment. The network archives the data for a daily time step.
		Continent: Americas
		Country: USA
		Stations: 50
		Status: running
		Data Range: from 1998-01-01  to 2010-12-30
		Type: project
		Url: https://hprcc.unl.edu/awdn/
		Reference: We acknowledge the work of Natalie Umphlett and the AWDN team in support of the ISMN;
		Variables: soil moisture, 
		Soil Moisture Depths: 0.10 - 0.10 m, 0.25 - 0.25 m, 0.50 - 0.50 m, 1.00 - 1.00 m
		Soil Moisture Sensors: ThetaProbe ML2X, Vitel, 

	BFG_Nw
		Abstract: 
		Continent: Europe
		Country: Germany
		Stations: 1
		Status: running
		Data Range: from 1986-11-01 
		Type: project
		Url: 
		Reference: We acknowledge the work of Jens Wilhelmi in support of the ISMN;
		Variables: air temperature, precipitation, soil moisture, soil temperature, surface temperature, 
		Soil Moisture Depths: 0.25 - 0.25 m, 0.50 - 0.66 m, 0.85 - 0.85 m, 1.05 - 1.05 m
		Soil Moisture Sensors: SMT100, TRIME-ES P2, 

	BIEBRZA_S-1
		Abstract: Preparation of the method for determining biomass and soil moisture changes on the basis of data delivered by recent satellite missions
		Continent: Europe
		Country: Poland
		Stations: 30
		Status: running
		Data Range: from 2015-09-15 
		Type: project
		Url: http://www.igik.edu.pl/en
		Reference: Dabrowska-Zielinska K., Musial J., Malinska A., Budzynska M., Gurdak R., Kiryla W., Bartold M., Grzybowski P., (2018), Soil Moisture in the Biebrza Wetlands Retrieved from Sentinel-1 Imagery. Remote Sensing 2018, Vol. 
10(12), 1979. https://doi.org/10.3390/rs10121979;

Musial, J. P., Dabrowska-Zielinska, K., Kiryla, W., Oleszczuk, R., Gnatowski, T. & Jaszczynski, J. (2016), ‘Derivation and validation of the high resolution satellite soil moisture products: a case study of the biebrza sentinel-1 validation sites’, Geoinformation Issues 8(1 (8)), 37–53. https://doi.org/10.34867/gi.2016.4;
		Variables: precipitation, soil temperature, soil moisture, air temperature, 
		Soil Moisture Depths: 0.05 - 0.05 m, 0.10 - 0.10 m, 0.20 - 0.20 m, 0.50 - 0.50 m
		Soil Moisture Sensors: GS-3, 

	BNZ-LTER
		Abstract: The Bonanza Creek (BNZ) LTER project was initiated in 1987 and since then has provided experimental and observational research designed to understand the dynamics, resilience, and vulnerability of Alaska"s boreal forest ecosystems. The project has illuminated the responses of boreal forest organisms and ecosystems to climate and various atmospheric inputs, focusing on forest and landscape dynamics and biogeochemistry. This project will continue that long-term line of research, expanding it to broaden the landscape under study, broaden the predictive realm of the resulting information, and to directly address the resilience of socio-economic systems. The project hypothesizes that the past observed high resilience of boreal ecosystems to interannual and decadal changes in environmental conditions is approaching a critical tipping point., 
This project contributes to understanding of the structure, function, and dynamics of boreal forest ecosystems and the broader boreal landscape, including the human communities. It assembles and integrates valuable long-term data sets on climate, hydrology, biology, ecology, and biogeochemical and geomorphic processes, as incorporates emerging data types, including molecular and social science data and digital images. The project has broad societal value through its contributions to knowledge that can inform management of boreal forest ecosystems and sustainability of subsistence communities. Its broader values also include extensive research-based training and educational program development. Its strong public outreach program includes collaborations between artists and scientists and strong linkages with Native organizations.
		Continent: Americas
		Country: Alaska
		Stations: 12
		Status: running
		Data Range: from 1989-05-01  to 2012-12-31
		Type: project
		Url: http://www.lter.uaf.edu/
		Reference: Van Cleve, Keith, Chapin, F.S. Stuart, Ruess, Roger W. 2015. Bonanza Creek Long Term Ecological Research Project Climate Database - University of Alaska Fairbanks.,
Bonanza Creek Long Term Ecological Research Project Climate Database. 2015. University of Alaska Fairbanks. https://www.lter.uaf.edu/;
		Variables: air temperature, precipitation, snow depth, snow water equivalent, soil moisture, soil temperature, surface temperature, 
		Soil Moisture Depths: 0.05 - 0.05 m, 0.10 - 0.10 m, 0.20 - 0.20 m, 0.50 - 0.50 m
		Soil Moisture Sensors: CS615, 

	CALABRIA
		Abstract: 
		Continent: Europe
		Country: Italy
		Stations: 5
		Status: running
		Data Range: from 2000-01-01 
		Type: meteo
		Url: http://www.cfd.calabria.it/
		Reference: Brocca, L., Hasenauer, S., Lacava, T., Melone, F., Moramarco, T., Wagner, W., A, D., Matgen, P., Mart´ınez-Fern´andez, J., Llorens, P., Latron, J., Martin, C. & Bittelli, M. (2011), ‘Soil moisture estimation through ascat and amsr-e sensors: An intercomparison and validation study across europe’, Remote Sensing of Environment 115, 3390–3408. https://doi.org/10.1016/j.rse.2011.08.003;
		Variables: air temperature, precipitation, soil moisture, 
		Soil Moisture Depths: 0.30 - 0.30 m, 0.60 - 0.60 m, 0.90 - 0.90 m
		Soil Moisture Sensors: ThetaProbe ML2X, 

	CAMPANIA
		Abstract: This italian data set consists of two stations located near to the city of Naples in the south of Italy. It is managed by the "Centro Funzionale per la Previsione Meteorologica e il Monitoraggio Meteo-Pluvio-Idrometrico e delle Frane". The ISMN contains data from the operational start in 2000 until the end of 2008. The data sets include soil moisture measured at a depth of 0.30 m, precipitation, and air temperature.
		Continent: Europe
		Country: Italy
		Stations: 2
		Status: inactive
		Data Range: from 2000-11-26  to 2008-12-31
		Type: project
		Url: http://www.regione.campania.it/
		Reference: Brocca, L., Hasenauer, S., Lacava, T., Melone, F., Moramarco, T., Wagner, W., A, D., Matgen, P., Mart´ınez-Fern´andez, J., Llorens, P., Latron, J., Martin, C. & Bittelli, M. (2011), ‘Soil moisture estimation through ascat and amsr-e sensors: An intercomparison and validation study across europe’, Remote Sensing of Environment 115, 3390–3408. https://doi.org/10.1016/j.rse.2011.08.003;
		Variables: air temperature, precipitation, soil moisture, 
		Soil Moisture Depths: 0.30 - 0.30 m
		Soil Moisture Sensors: ThetaProbe ML2X, 

	CHINA
		Abstract: Soil moisture datasets for 40 stations were collected from the Institute of Geographical Sciences and Natural Resources Research at the Chinese Academy of Sciences. The soil moisture observations were taken using the gravimetric technique and were converted to total soil moisture [cm]. After downloading the measurements from Alan Robock"s "Global Soil Moisture Database" the data was converted into volumetric soil moisture [m^3/m^3]. The dataset provides observations three times a month from 1981 to 1991 in 11 different layers.
		Continent: Asia
		Country: China
		Stations: 40
		Status: inactive
		Data Range: from 1981-01-08  to 1999-12-28
		Type: project
		Url: 
		Reference: Liu S., Mo X, Li H., Peng G., Robock A. 2001 The spatial variation of soil moisture in China:Geostatistical Characteristics. Journal of the Meteorological Society of Japan , 79 (2B) 555-574, https://doi.org/10.2151/jmsj.79.555;
Robock, Alan, Konstantin Y. Vinnikov, Govindarajalu Srinivasan, Jared K. Entin, Steven E. Hollinger, Nina A. Speranskaya, Suxia Liu, and A. Namkhai, 2000: The Global Soil Moisture Data Bank. Bull. Amer. Meteorol. Soc., 81, 1281-1299, https://doi.org/10.1175/1520-0477(2000)081<1281:TGSMDB>2.3.CO,2;
		Variables: soil moisture, 
		Soil Moisture Depths: 0.00 - 0.05 m, 0.05 - 0.10 m, 0.10 - 0.20 m, 0.20 - 0.30 m, 0.30 - 0.40 m, 0.40 - 0.50 m, 0.50 - 0.60 m, 0.60 - 0.70 m, 0.70 - 0.80 m, 0.80 - 0.90 m, 0.90 - 1.00 m
		Soil Moisture Sensors: Coring device-auger, 

	COSMOS
		Abstract: A new project to measure soil moisture using a cosmic-ray technique. Currently, there are 67 stations deployed in 7 countries, which 59 are in USA, 1 in Germany, 1 in Switzerland, 1 in France, 1 in Brasil, 2 in Kenya, 1 in United Kingdom and 1 in Mexico.
		Continent: Americas
		Country: USA
		Stations: 109
		Status: running
		Data Range: from 2008-04-28 
		Type: project
		Url: http://cosmos.hwr.arizona.edu/
		Reference: Zreda M., Desilets D., Ferré Ty P.A., Scott R.L., “Measuring soil moisture content non-invasively at intermediate spatial scale using cosmic-ray neutrons”, Geophysical Research Letters 35(21), 2008. https://doi.org/10.1029/2008GL035655;

Zreda, M., W.J. Shuttleworth, X. Zeng, C. Zweck, D. Desilets, T. Franz, and R. Rosolem, 2012. COSMOS: the COsmic-ray Soil Moisture Observing System. Hydrology and Earth System Sciences 16, 4079-4099, https://doi.org/10.5194/hess-16-4079-2012;
		Variables: soil moisture, 
		Soil Moisture Depths: 0.00 - 0.04 m, 0.00 - 0.05 m, 0.00 - 0.06 m, 0.00 - 0.07 m, 0.00 - 0.08 m, 0.00 - 0.09 m, 0.00 - 0.10 m, 0.00 - 0.11 m, 0.00 - 0.12 m, 0.00 - 0.13 m, 0.00 - 0.14 m, 0.00 - 0.15 m, 0.00 - 0.16 m, 0.00 - 0.17 m, 0.00 - 0.18 m, 0.00 - 0.19 m, 0.00 - 0.20 m, 0.00 - 0.21 m, 0.00 - 0.22 m, 0.00 - 0.23 m, 0.00 - 0.24 m, 0.00 - 0.25 m, 0.00 - 0.26 m, 0.00 - 0.27 m, 0.00 - 0.28 m, 0.00 - 0.29 m, 0.00 - 0.30 m, 0.00 - 0.31 m, 0.00 - 0.32 m, 0.00 - 0.33 m, 0.00 - 0.34 m, 0.00 - 0.36 m, 0.00 - 0.37 m, 0.00 - 0.38 m, 0.00 - 0.39 m, 0.00 - 0.40 m, 0.00 - 0.41 m, 0.00 - 0.44 m, 0.00 - 0.55 m, 0.00 - 0.69 m, 0.00 - 0.90 m
		Soil Moisture Sensors: Cosmic-ray Probe, 

	COSMOS-UK
		Abstract: The COSMOS-UK network is established and operated by the UK Centre for Ecology and Hydrology (UKCEH). It is the first network to systematically measure soil moisture throughout the UK, and represents a range of climate, soils and vegetation. The network consists of ~50 sites, each recording a number of hydrometeorological and soil variables. Each site hosts a cosmic-ray neutron sensor (CRNS); a novel sensor technology which counts fast neutrons in the surrounding atmosphere. In combination with the recorded hydrometeorological data, neutron counts are used to derive Volumetric Water Content (VWC) over a field scale (~0.1 km2) (COSMOS VWC), at daily resolution. Alongside the CRNS, each site hosts several point soil moisture probes recording at 30 minute resolution. More information is available from the COSMOS-UK website: https://cosmos.ceh.ac.uk/ . The full dataset, along with quality flags, is available at: https://doi.org/10.5285/5060cc27-0b5b-471b-86eb-71f96da0c80f .


************************************************************************************************************
IMPORTANT NOTE: The Cosmic-ray neutron sensor (CRNS) is placed just above the ground and counts the naturally occurring neutrons (originating from cosmic rays). The neutrons are scattered by hydrogen atoms, which are present primarily in the soil water. The CRNS neutron count rate (after corrections) reduces with increasing soil moisture content, and can therefore be used, via a calibration curve, to infer the soil Volumetric Water Content (VWC). The CRNS counts neutrons which may have been scattered many times, such that they may have interacted with the ground soil moisture over distances of more than 200 m from the probe, and from tens centimetres of soil depth. This local scattering of neutrons determines the horizontal and vertical footprint of the sensor, or the measurement support volume, and yields the important characteristic of averaging the soil moisture measurement over a large area. Using neutron scattering models, the typical footprint is determined (~12 hectares), although there is some dependence on the VWC, particularly for the soil depth of measurement. For this reason, the CRNS depth of measurement is given as an average for the time series and location, but it should be appreciated that actual sensing depth will be reduced in wet conditions, compared with dry conditions (from around 10 cm to 30 cm depth respectively).
************************************************************************************************************

		Continent: Europe
		Country: UK
		Stations: 49
		Status: running
		Data Range: from 2013-10-01 
		Type: project
		Url: https://cosmos.ceh.ac.uk/
		Reference: Cooper, H. M., Bennett, E., Blake, J., Blyth, E., Boorman, D., Cooper, E., Evans, J., Fry, M., Jenkins, A., Morrison, R., Rylett, D., Stanley, S., Szczykulska, M., Trill, E., Antoniou, V., Askquith-Ellis, A., Ball, L., Brooks, M., Clarke, M. A., Cowan, N., Cumming, A., Farrand, P., Hitt, O., Lord, W., Scarlett, P., Swain, O., Thornton, J., Warwick, A., and Winterbourn, B.: COSMOS-UK: national soil moisture and hydrometeorology data for environmental science research, Earth Syst. Sci. Data, 13, 1737–1757, https://doi.org/10.5194/essd-13-1737-2021, 2021.
		Variables: precipitation, soil temperature, air temperature, soil moisture, snow depth, 
		Soil Moisture Depths: 0.00 - 0.30 m, 0.05 - 0.05 m, 0.10 - 0.10 m, 0.15 - 0.15 m, 0.25 - 0.25 m, 0.50 - 0.50 m
		Soil Moisture Sensors: Cosmic-ray Probe, TDT, 

	CTP_SMTMN
		Abstract: A dense monitoring network that consists of 56 stations is established on the central Tibetan Plateau to measure two state variables (soil moisture and temperature) at three spatial scales (1.0, 0.3, 0.1 degree) and four soil depths (0~5, 10, 20, and 40 cm). Elevations of these stations vary over 4470~4950 m. The experimental area is characterized by low biomass, high soil moisture dynamic range, and typical freeze-thaw cycle. As auxiliary parameters of this network, soil texture and soil organic carbon content are measured at each station to support further studies. All the sensors have been calibrated by taking account of the impact of soil texture and soil organic carbon content on the measurements. 
As the highest soil moisture network above sea level in the world, this network meets the requirement for evaluating a variety of soil moisture products and for soil moisture scaling analyses. 
Note that, 
a)	some of the stations are shared by two or three sub-scale networks (indicated by the station names) and thus the number of station names is up to 69 in total; 
b)	in order to ensure the continuous measurements of the near surface (0~5 cm) SM/TM, the original surface layer sensor (when get damaged) is usually replaced with another sensor at deeper depth during field maintenance. Therefore there may be offset in a time series before and after the replacing time point (shown in Table 1). 
Table 1. Sensor replacing/exchanging time points at specific stations. 
Station_Name	0-0.05 m	0.40 m
M19	        10/13/2012	
L01	          6/14/2012	  6/14/2012
L07_M13	  6/15/2012	  6/15/2012
L35	        10/15/2012	10/15/2012
M03	        10/13/2012	10/13/2012
M07_S01	10/13/2012	10/13/2012

		Continent: Asia
		Country: China
		Stations: 57
		Status: running
		Data Range: from 2007-07-01 
		Type: project
		Url: http://dam.itpcas.ac.cn/rs/?q=data#CTP-SMTMN
		Reference: Yang, K., J. Qin, L. Zhao, Y. Y. Chen, W. J. Tang, M. L. Han, Lazhu, Z. Q. Chen, N. Lv, B. H. Ding, H. Wu, C. G. Lin, 2013. A Multi-Scale Soil Moisture and Freeze-Thaw Monitoring Network on the Third Pole, Bulletin of the American Meteorological Society, https://doi.org/10.1175/BAMS-D-12-00203.1;
		Variables: soil temperature, soil moisture, 
		Soil Moisture Depths: 0.00 - 0.05 m, 0.10 - 0.10 m, 0.20 - 0.20 m, 0.40 - 0.40 m
		Soil Moisture Sensors: EC-TM, 5TM, 

	CW3E
		Abstract: Research lab
		Continent: Americas
		Country: USA
		Stations: 24
		Status: running
		Data Range: from 2004-08-01 
		Type: Hydroclimate research
		Url: https://cw3e.ucsd.edu/
		Reference: Ralph, F. Martin; Wilson, Anna M.; Alden, Douglas; Ellis, Carolyn J.; Cooper, Ava; Paulsson, Kerstin; Kawzenuk, Brian K.; Yao, Peter (2022). Center for Western Weather and Water Extremes (CW3E) Surface Meteorological Observational Dataset. UC San Diego Library Digital Collections. https://doi.org/10.6075/J0SX6DDH
		Variables: precipitation, soil temperature, air temperature, soil moisture, 
		Soil Moisture Depths: 0.05 - 0.05 m, 0.05 - 0.05 m, 0.10 - 0.10 m, 0.10 - 0.10 m, 0.15 - 0.15 m, 0.15 - 0.15 m, 0.20 - 0.20 m, 0.20 - 0.20 m, 0.50 - 0.50 m, 0.51 - 0.51 m, 1.00 - 1.00 m, 1.02 - 1.02 m
		Soil Moisture Sensors: CS616, Stevens Hydra Probe, 

	DAHRA
		Abstract: Earth Observation of long term changes in land surface moisture conditions
		Continent: Africa
		Country: Senegal
		Stations: 1
		Status: running
		Data Range: from 2002-07-04 
		Type: project
		Url: https://ign.ku.dk/english/research/geography/environment-society-developing-countries/
		Reference: Tagesson, T., Fensholt, R., Guiro, I., Rasmussen, M., Huber, S., Mbow, C., Garcia, M., Horion, S., Sandholt, I., Holm-Rasmussen, B., Göttsche, F.-M., Ridler, M., Boke-Ol´en, N., Olsen, J., Ehammer, A., Madsen, M., Olesen, F. & Ardö, J. (2014), ‘Ecosystem properties of semi-arid savanna grassland in west africa and its relationship to environmental variability’, Global Change Biology 21., https://doi.org/10.1111/gcb.12734;
		Variables: air temperature, precipitation, soil moisture, soil temperature, 
		Soil Moisture Depths: 0.05 - 0.05 m, 0.10 - 0.10 m, 0.30 - 0.30 m, 0.50 - 0.50 m, 1.00 - 1.00 m
		Soil Moisture Sensors: ThetaProbe ML2X, 

	FLUXNET-AMERIFLUX
		Abstract: Datasets of 2 stations near the city of Sacramento are provided. They are managed by Dennis D. Baldocchi, University of California, Berkeley. 
		Continent: Americas
		Country: USA
		Stations: 8
		Status: running
		Data Range: from 2000-10-22 
		Type: project
		Url: http://ameriflux.lbl.gov/
		Reference: We acknowledge the work of Dennis D. Baldocchi and the FLUXNET-AMERIFLUX team in support of the ISMN;
		Variables: air temperature, precipitation, soil moisture, soil temperature, 
		Soil Moisture Depths: 0.00 - 0.00 m, 0.00 - 0.15 m, 0.02 - 0.02 m, 0.05 - 0.05 m, 0.10 - 0.10 m, 0.15 - 0.30 m, 0.20 - 0.20 m, 0.30 - 0.45 m, 0.45 - 0.60 m, 0.50 - 0.50 m
		Soil Moisture Sensors: CS655, Moisture Point PRB-K, ThetaProbe ML2X, ThetaProbe ML3, 

	FMI
		Abstract: The finnish network "FMI" includes one station that contains multiple soil moisture measurements in 2cm and 10cm depth, taken only a few meters next to each other. Additionaly air temperature is measured in 2m height and soil temperature in 2cm depth. The datasets start at 2007-01-25 and are updated once a day. The FMI is the first network that has been implemented with data updates in Near Real Time.
		Continent: Europe
		Country: Finland
		Stations: 27
		Status: running
		Data Range: from 2007-01-25 
		Type: project
		Url: http://fmiarc.fmi.fi/
		Reference: Ikonen, J., Smolander, T., Rautiainen, K., Cohen, J., Lemmetyinen, J., Salminen, M. & Pulliainen, J. (2018), ‘Spatially distributed evaluation of esa cci soil moisture products in a northern boreal forest environment’, Geosciences 8(2), 51., https://doi.org/10.3390/geosciences8020051;

Ikonen, J., Vehviläinen, J., Rautiainen, K., Smolander, T., Lemmetyinen, J., Bircher, S. & Pulliainen, J. (2015), ‘The sodankylä in-situ soil moisture observation network: an example application to earth observation data product evaluation’, GID 5(2), 599–629., https://doi.org/10.5194/gi-5-95-2016;
		Variables: soil temperature, air temperature, soil moisture, 
		Soil Moisture Depths: 0.02 - 0.02 m, 0.05 - 0.05 m, 0.10 - 0.10 m, 0.20 - 0.20 m, 0.40 - 0.40 m, 0.60 - 0.60 m, 0.80 - 0.80 m
		Soil Moisture Sensors: 5TE, CS655, ThetaProbe ML2X, 

	FR_Aqui
		Abstract: The Fr_Aqui network is located in France and hosted by the Institue of Agricultural Research (INRA); it consists of 5 stations with soil moisture and soil temperature measurements in 6 different  depths.
		Continent: Europe
		Country: France
		Stations: 5
		Status: running
		Data Range: from 2010-01-01 
		Type: meteo
		Url: 
		Reference: Al-Yaari, A., Dayau, S., Chipeaux, C., Aluome, C., Kruszewski, A., Loustau, D. & Wigneron, J.-P. (2018), ‘The aqui soil moisture network for satellite microwave remote sensing validation in south-western france’, Remote Sensing 10(11), https://doi.org/10.3390/rs10111839;

Wigneron, J.-P., Dayan, S., Kruszewski, A., Aluome, C., Al-Yaari, A., Fan, L., Guven, S., Chipeaux, C., Moisy, C., Guyon, D. & Loustau, D. (2018), The aqui network: Soil moisture sites in the “les landes” forest and graves vineyards (bordeaux aquitaine region, france), pp. 3739–3742., https://doi.org/10.1109/IGARSS.2018.8517392;
		Variables: soil moisture, soil temperature, 
		Soil Moisture Depths: 0.01 - 0.01 m, 0.03 - 0.03 m, 0.05 - 0.05 m, 0.10 - 0.10 m, 0.15 - 0.15 m, 0.20 - 0.20 m, 0.21 - 0.21 m, 0.25 - 0.25 m, 0.30 - 0.30 m, 0.34 - 0.34 m, 0.45 - 0.45 m, 0.50 - 0.50 m, 0.55 - 0.55 m, 0.56 - 0.56 m, 0.70 - 0.70 m, 0.80 - 0.80 m, 0.90 - 0.90 m
		Soil Moisture Sensors: ThetaProbe ML2X, 

	GROW
		Abstract: The GROW Observatory (GROW) is a European-wide project engaging thousands of growers, scientists and others passionate about the land. We will discover together, using simple tools to better manage soil and grow food, while contributing to vital scientific environmental monitoring.
		Continent: Europe
		Country: UK
		Stations: 151
		Status: inactive
		Data Range: from 2016-11-01  to 2019-10-31
		Type: project
		Url: https://growobservatory.org/index.html
		Reference: Xaver, A., Zappa, L., Rab, G., Pfeil, I., Vreugdenhil, M., Hemment, D. & Dorigo, W. A. (2020), ‘Evaluating the suitability of the consumer low-cost parrot ﬂower power soil moisture sensor for scientiﬁc environmental applications’, Geoscientiﬁc Instrumentation, Methods and Data Systems 9(1), 117–139., https://doi.org/10.5194/gi-9-117-2020;

Zappa, L., Woods, M., Hemment, D., Xaver, A. & Dorigo, W. (2020), Evaluation of remotely sensed soil moisture products using crowdsourced measurements, Eighth International Conference on Remote Sensing and Geoinformation of Environment, SPIE, Cyprus., https://doi.org/;

Zappa, L., Forkel, M., Xaver, A. & Dorigo, W. (2019), ‘Deriving ﬁeld scale soil moisture from satellite observations and ground measurements in a hilly agricultural region’, Remote Sensing 11(22), 2596., https://doi.org/10.3390/rs11222596, https://doi.org/10.1117/12.2571913;
		Variables: air temperature, soil moisture, 
		Soil Moisture Depths: 0.00 - 0.10 m
		Soil Moisture Sensors: Flower Power, 

	GTK
		Abstract: Geological Survey of Finland
		Continent: Europe
		Country: Finland
		Stations: 7
		Status: running
		Data Range: from 2001-03-01 
		Type: project
		Url: https://www.gtk.fi/en/
		Reference: We acknowledge the work of Raimo Sutinen and the GTK team in support of the ISMN;
		Variables: air temperature, soil moisture, soil temperature, 
		Soil Moisture Depths: 0.10 - 0.10 m, 0.30 - 0.30 m, 0.50 - 0.50 m, 0.70 - 0.70 m, 0.90 - 0.90 m
		Soil Moisture Sensors: CS615, CS616, 

	HiWATER_EHWSN
		Abstract: The HiWATER EHWSN network is located at farmland in the middle stream of the Heihe River Basin, China. It consists of stations 174.  The datasets are  available from April 2012 till  September 2012 and are collected by the Cold and Arid Regions Environmental and Engineering Research Institute (CAREERI), Chineses Academy of Science (CAS).
		Continent: Asia
		Country: China
		Stations: 174
		Status: inactive
		Data Range: from 2012-05-12  to 2012-09-20
		Type: campaign
		Url: http://www.westgis.ac.cn/
		Reference: Jian Kang, Xin Li, Rui Jin, Yong Ge, Jinfeng Wang and Jianghao Wang. Hybrid optimal design of the eco-hydrological wireless sensor network in the middle reach of the Heihe River Basin, China, Sensors, 2014, 14(10): 19095~19114, https://doi.org/10.3390/s141019095;

Rui Jin, Xin Li, Baoping Yan, Xiuhong Li, Wanmin Luo, Mingguo Ma, Jianwen Guo, Jian Kang, Zhongli Zhu and Shaojie Zhao. A nested ecohydrological wireless sensor network for capturing the surface heterogeneity in the midstream areas of the Heihe River Basin, China，IEEE Geoscience and Remote Sensing Letters, 2014, 11(11): 2015~2019, https://doi.org/10.1109/LGRS.2014.2319085;
		Variables: soil temperature, soil moisture, 
		Soil Moisture Depths: 0.04 - 0.04 m, 0.10 - 0.10 m, 0.20 - 0.20 m, 0.40 - 0.40 m
		Soil Moisture Sensors: FM100, SPADE Time Domain Transmissivity, Hydraprobe II, 

	HOAL
		Abstract: 
		Continent: Europe
		Country: Austria
		Stations: 33
		Status: running
		Data Range: from 2013-12-07 
		Type: research network

		Url: https://hoal.hydrology.at/research/soil-moisture

		Reference: Blöschl, Günter, et al. "The hydrological open air laboratory (HOAL) in Petzenkirchen: A hypothesis-driven observatory." Hydrology and Earth System Sciences 20.1 (2016): 227-255., https://doi.org/10.5194/hess-20-227-2016;

M. Vreugdenhil et al., "Towards a high-density soil moisture network for the validation of SMAP in Petzenkirchen, Austria," 2013 IEEE International Geoscience and Remote Sensing Symposium - IGARSS, Melbourne, VIC, 2013, pp. 1865-1868, https://doi.org/10.1109/IGARSS.2013.6723166.;
		Variables: soil temperature, soil moisture, 
		Soil Moisture Depths: 0.05 - 0.05 m, 0.10 - 0.10 m, 0.20 - 0.20 m, 0.50 - 0.50 m, 1.00 - 1.00 m
		Soil Moisture Sensors: SPADE Time Domain Transmissivity, 

	HOBE
		Abstract: Soil moisture and soil temperature network with 30 stations within the area of major signal contribution of one selected SMOS grid node in the Skjern River Catchment
		Continent: Europe
		Country: Denmark
		Stations: 32
		Status: inactive
		Data Range: from 2009-09-08  to 2019-03-13
		Type: project
		Url: http://www.hobe.dk/
		Reference: Jensen, K. & Refsgaard, J. (2018), ‘Hobe: The danish hydrological observatory’, Vadose Zone Journal 17., https://doi.org/10.2136/vzj2018.03.0059;

Bircher, S., Skou, N., Jensen, K., Walker, J. & Rasmussen, L. (2012), ‘A soil moisture and temperature network for smos validation in western denmark’, Hydrology and Earth System Sciences 16., https://doi.org/10.5194/hess-16-1445-2012;
		Variables: soil temperature, soil moisture, 
		Soil Moisture Depths: 0.00 - 0.05 m, 0.20 - 0.25 m, 0.50 - 0.55 m
		Soil Moisture Sensors: Decagon 5TE, 

	HSC_SEOLMACHEON
		Abstract: 
		Continent: Asia
		Country: Korea
		Stations: 1
		Status: running
		Data Range: from 2008-01-01 
		Type: project
		Url: http://www.hsc.re.kr
		Reference: We acknowledge the work of Yeongil Lee and Minha Choi and the HSC_SEOLMACHEON team in support of the ISMN;
		Variables: soil moisture, 
		Soil Moisture Depths: 0.00 - 0.10 m
		Soil Moisture Sensors: Hydraprobe Analog (CR800), 

	HYDROL-NET_PERUGIA
		Abstract: This network has 1 station called WEEF (Water Engineering Experimental Field) near the city of Perugia, Central Italy. It provides of measurements of soil moisture at 4 different depths, soil temperature at 2 depths, precipitation, and air temperature. It is operated by the Department of Civil and Environmental Engineering of the University of Perugia since the beginning of 2010.
		Continent: Europe
		Country: Italy
		Stations: 2
		Status: running
		Data Range: from 2010-01-01 
		Type: project
		Url: http://www.ing1.unipg.it/
		Reference: FLAMMINI A., MORBIDELLI R., SALTALIPPI C., PICCIAFUOCO T., CORRADINI C., GOVINDARAJU R.S. “Reassessment of a semi-analytical field-scale infiltration model through experiments under natural rainfall events”, Journal of Hydrology, 565, 835-845, 2018, Codice Scopus: 2-s2.0-85053026749, Codice WOS: 000447477200069, https://doi.org/10.1016/j.jhydrol.2018.08.073;

FLAMMINI A., CORRADINI C., MORBIDELLI R., SALTALIPPI C., PICCIAFUOCO T., GIRALDEZ J.V. “Experimental Analyses of the Evaporation Dynamics in Bare Soils under Natural Conditions”, Water Resources Management, 32 (3), 1153-1166, 2018, Codice Scopus: 2-s2.0-85034632795, Codice WOS: 000422982300020, https://doi.org/10.1007/s11269-017-1860-x;

Morbidelli, R., Saltalippi, C., Flammini, A., Cifrodelli, M., Picciafuoco, T., Corradini, C. & Govindaraju, R. S. (2017), ‘In situ measurements of soil saturated hydraulic conductivity: Assessment of reliability through rainfall–runoﬀ experiments’, Hydrological Processes 31(17), 3084–3094, https://doi.org/10.1002/hyp.11247;

MORBIDELLI R., SALTALIPPI C., FLAMMINI A., ROSSI E., CORRADINI C., “Soil water content vertical profiles under natural conditions: matching of experiments and simulations by a conceptual model”, Hydrological Processes, 28 (17), 4732-4742, 2014, Codice Scopus: 2-s2.0-84905732846, Codice WOS: 000340611500006, https://doi.org/10.1002/hyp.9973;

MORBIDELLI R., CORRADINI C., SALTALIPPI C., FLAMMINI A., ROSSI E., “Infiltration-soil moisture redistribution under natural conditions: experimental evidence as a guideline for realizing simulation models”, Hydrology and Earth System Sciences, 15, 2937-3945, 2011, Codice Scopus: 2-s2.0-80052885350, Codice WOS: 000295357100012, https://doi.org/10.5194/hess-15-2937-2011;
		Variables: air temperature, precipitation, soil moisture, soil temperature, 
		Soil Moisture Depths: 0.05 - 0.05 m, 0.15 - 0.15 m, 0.25 - 0.25 m, 0.35 - 0.35 m
		Soil Moisture Sensors: TDR-Soil Moisture Equipment Corp. TRASE-BE, 

	ICN
		Abstract: "Soil moisture datasets for 19 stations, collected and processed by the "Illinois State Water Survey". The soil moisture data given in total soil moisture [mm] for 11 layers of each station were converted to volumetric soil moisture [m^3/m^3]
"
		Continent: Americas
		Country: USA
		Stations: 19
		Status: inactive
		Data Range: from 1981-01-01  to 2010-11-21
		Type: project
		Url: http://www.isws.illinois.edu/warm/
		Reference: Hollinger, Steven E., and Scott A. Isard, 1994: A soil moisture climatology of Illinois. J. Climate, 7, 822-833., https://doi.org/10.1175/1520-0442(1994)007<0822:ASMCOI>2.0.CO,2;
		Variables: precipitation, soil moisture, soil temperature, 
		Soil Moisture Depths: 0.00 - 0.10 m, 0.10 - 0.30 m, 0.30 - 0.50 m, 0.50 - 0.70 m, 0.70 - 0.90 m, 0.90 - 1.10 m, 1.10 - 1.30 m, 1.30 - 1.50 m, 1.50 - 1.70 m, 1.70 - 1.90 m, 1.90 - 2.00 m
		Soil Moisture Sensors: Stevens Hydra Probe, Troxler Neutron Depth Probe, Troxler Neutron Surface Probe, 

	IIT_KANPUR
		Abstract: The station called AIRSTRIP and located in thenord-center of India provide Soil Moisture datasets at 4 depths. This station is operated by the Indian Institute of Technology Kanpur.
		Continent: Asia
		Country: India
		Stations: 1
		Status: running
		Data Range: from 2011-06-16 
		Type: project
		Url: http://www.iitk.ac.in/
		Reference: We acknowledge the work of Tripathi Shivam and the IIT_Kanpur team in support of the ISMN;
		Variables: soil moisture, 
		Soil Moisture Depths: 0.10 - 0.10 m, 0.25 - 0.25 m, 0.50 - 0.50 m, 0.80 - 0.80 m
		Soil Moisture Sensors: WaterScout SM100, 

	IMA_CAN1
		Abstract: Soil Moisture Network installed in three vineyard plots, with different inter-rows soil management. The plots are also monitored for runoff and soil erosion.Weather data available from a station near the plots.
		Continent: Europe
		Country: Italy
		Stations: 12
		Status: inactive
		Data Range: from 2011-08-23  to 2015-10-23
		Type: campaign
		Url: http://sustag.imamoter.cnr.it/index.php/cannona-db
		Reference: Capello, G., Biddoccu, M., Ferraris, S. & Cavallo, E. (2019), ‘Effects of tractor passes on hydrological and soil erosion processes in tilled and grassed vineyards’, Water 11(10), 2118., https://doi.org/10.3390/w11102118;

Raffelli, G., Id, M., Previati, M., Id, D., Canone, D., Gisolo, D., Bevilacqua, I., Capello, G., Biddoccu, M., Cavallo, E., Id, R., Deiana, R., Id, G., Cassiani, G. & Ferraris, S. (2018), ‘Local-and plot-scale measurements of soil moisture: Time and spatially resolved ﬁeld techniques in plain, hill and mountain sites’, Water 9., https://doi.org/10.3390/w9090706;

Marcella Biddoccu, Stefano Ferraris, Francesca Opsi, Eugenio Cavallo. Long-term monitoring of soil management effects on runoff and soil erosion in sloping vineyards in Alto Monferrato (North–West Italy). Soil and Tillage Research, Volume 155, 2016, Pages 176-189, ISSN 0167-1987, https://doi.org/10.1016/j.still.2015.07.005.;
		Variables: soil moisture, soil temperature, 
		Soil Moisture Depths: 0.10 - 0.10 m
		Soil Moisture Sensors: 5TM, 

	IOWA
		Abstract: Soil moisture measurements for two different catchments with three sites per catchment were taken in the southwestern part of Iowa. Soil moisture observations during the period 1972 to 1994 are available. Measurements were taken at 12 different soil layers up to a depth of 2.6 m. The gravimetric method was used for the top layers and a neutron probe for the lower layers. On average observations were made twice a month between April and October.
		Continent: Americas
		Country: USA
		Stations: 6
		Status: inactive
		Data Range: from 1972-04-04  to 1994-11-15
		Type: project
		Url: 
		Reference: Robock, Alan, Konstantin Y. Vinnikov, Govindarajalu Srinivasan, Jared K. Entin, Steven E. Hollinger, Nina A. Speranskaya, Suxia Liu, and A. Namkhai, 2000: The Global Soil Moisture Data Bank. Bull. Amer. Meteorol. Soc., 81, 1281-1299, https://doi.org/10.1175/1520-0477(2000)081<1281:TGSMDB>2.3.CO,2;
		Variables: soil moisture, 
		Soil Moisture Depths: 0.00 - 0.08 m, 0.08 - 0.15 m, 0.15 - 0.30 m, 0.30 - 0.46 m, 0.46 - 0.69 m, 0.69 - 0.84 m, 0.84 - 1.07 m, 1.07 - 1.37 m, 1.37 - 1.68 m, 1.68 - 1.98 m, 1.98 - 2.29 m, 2.29 - 2.59 m
		Soil Moisture Sensors: n.s., 

	IPE
		Abstract: 
		Continent: Europe
		Country: Spain
		Stations: 2
		Status: running
		Data Range: from 2008-04-03 
		Type: meteo
		Url: 
		Reference: Alday, J. G., Camarero, J. J., Revilla, J. & Dios, V. R. (2020), ‘Similar diurnal, seasonal and annual rhythms in radial root expansion across two coexisting mediterranean oak species’, Tree Physiology, https://doi.org/10.1093/treephys/tpaa041;
		Variables: air temperature, precipitation, soil moisture, soil temperature, 
		Soil Moisture Depths: 0.00 - 0.06 m, 0.00 - 0.30 m, 0.10 - 0.10 m
		Soil Moisture Sensors: CS616, CS650, ThetaProbe ML2X, 

	iRON
		Abstract: 
		Continent: Americas
		Country: USA
		Stations: 10
		Status: running
		Data Range: from 2012-06-07 
		Type: meteo
		Url: https://agci.org/iron
		Reference: Osenga, E.C., Vano, J.A., and Arnott, J.C. 2021. A community-supported weather and soil moisture monitoring database of the Roaring Fork catchment of the Colorado River Headwaters. Hydrologic Processes, 2021:35. https://doi.org/10.1002/hyp.14081;

Osenga, E. C., Arnott, J. C., Endsley, K. A., & Katzenberger, J. W. (2019). Bioclimatic and soil moisture monitoring across elevation in a mountain watershed: Opportunities for research and resource management. Water Resources Research, 55. https://doi.org/10.1029/2018WR023653;
		Variables: air temperature, precipitation, soil moisture, soil temperature, 
		Soil Moisture Depths: 0.05 - 0.05 m, 0.20 - 0.20 m, 0.50 - 0.50 m, 0.51 - 0.51 m, 1.00 - 1.00 m
		Soil Moisture Sensors: 10HS, EC5, EC5 I, EC5 II, HMP155, Hydraprobe II, 

	KHOREZM
		Abstract: 
		Continent: Asia
		Country: Uzbekistan
		Stations: 7
		Status: inactive
		Data Range: from 2010-01-01  to 2011-12-31
		Type: project
		Url: 
		Reference: We acknowledge the work of Patrick Knöfel and the KHOREZM team in support of the ISMN;
		Variables: air temperature, soil moisture, soil temperature, surface temperature, 
		Soil Moisture Depths: 0.00 - 0.05 m
		Soil Moisture Sensors: ThetaProbe ML2X, 

	KIHS_CMC
		Abstract: 
		Continent: Asia
		Country: Korea
		Stations: 18
		Status: running
		Data Range: from 2019-03-20 
		Type: project
		Url: http://kihs.re.kr
		Reference: We acknowledge the work of Kiyoung Kim, Yeongil Lee and the KIHS_CMC, KIHS_SMC teams in support of the ISMN:
		Variables: soil moisture, 
		Soil Moisture Depths: 0.10 - 0.10 m, 0.30 - 0.30 m, 0.40 - 0.40 m, 0.50 - 0.50 m, 0.60 - 0.60 m, 0.90 - 0.90 m
		Soil Moisture Sensors: Buriable Waveguide, 

	KIHS_SMC
		Abstract: 
		Continent: Asia
		Country: Korea
		Stations: 19
		Status: running
		Data Range: from 2019-03-21 
		Type: campaign
		Url: http://kihs.re.kr
		Reference: We acknowledge the work of Kiyoung Kim, Yeongil Lee and the KIHS_CMC, KIHS_SMC teams in support of the ISMN:
		Variables: soil moisture, 
		Soil Moisture Depths: 0.10 - 0.10 m, 0.20 - 0.20 m, 0.30 - 0.30 m, 0.40 - 0.40 m, 0.50 - 0.50 m, 0.60 - 0.60 m
		Soil Moisture Sensors: Buriable Waveguide, 

	LABFLUX
		Abstract: The Lab performs research, tutorial ad dissemination regarding measurements and modeling of water and moisture fluxes in the natural, agricultural and built environments. Main focus is on porous media like soil, snow and wood. Specifically addressed are the monitoring of : meteorological variables, energy fluxes, soil-plant-atmosphere interactions, water dynamics, and hillslope processes.
		Continent: Europe
		Country: Italy
		Stations: 4
		Status: running
		Data Range: from 2012-01-01 
		Type: project
		Url: https://www.dist.polito.it/il_dipartimento/laboratori/labflux
		Reference: We acknowledge the work of Davide Gisolo, Canone Davide, Ferrari Stefano, Alessio Gentile and Previati Maurizio as well as the LABFLUX network team in support of the ISMN;

		Variables: air temperature, precipitation, soil moisture, soil temperature, 
		Soil Moisture Depths: 0.10 - 0.10 m, 0.20 - 0.20 m, 0.40 - 0.40 m
		Soil Moisture Sensors: CS616, Meter EC-5, 

	LAB-net
		Abstract: In Chile, remote sensing applications related to soil moisture and evapotranspiration estimates have increased during the last decades because of the drought and the water use conflicts which generate a strong interest on water demand. To address these problems on water balance in large scales by using remote sensing imagery, LAB-net was created as the first soil moisture network located in Chile over four different land cover types. These land cover types are vineyard and olive orchards located in a semi-arid region of Copiapó Valley, a well irrigated raspberry crop located in the central zone of of Tinguiririca Valley and a green grass rangeland located Austral zone of Chile. Over each site, a well implemented meteorological station is continuously measuring a 5 minute intervals above the following parameters: soil moisture and temperature at two ground levels (5 and 20 cm), air temperature and relative humidity, net radiation, global radiation, brightness surface temperature (8 – 14 µm), rainfall and ground fluxes. This is the first approach of an integrated soil moisture network in Chile. The data generated by this network is freely available for any research or scientific purpose related to current and future soil moisture satellite missions.    

		Continent: Americas
		Country: Chile
		Stations: 4
		Status: running
		Data Range: from 2014-07-18 
		Type: meteo
		Url: http://www.biosfera.uchile.cl/LAB-net.html

		Reference: Mattar, C., Santamaria-Artigas, A., Duran-Alarcon, C., Olivera-Guerra, L., Fuster, R. & Borvar´an, D. (2016), ‘The lab-net soil moisture network: Application to thermal remote sensing and surface energy balance’, Data 1(1), https://doi.org/10.3390/data1010006;

Mattar, C., Santamaría-Artigas, A., Durán-Alarcón, C., Olivera-Guerra, L. and Fuster, R., 2014, September. LAB-net the first Chilean soil moisture network for remote sensing applications. In Quantitative Remote Sensing Symposium (RAQRS) (pp. 22-26);
		Variables: air temperature, precipitation, soil moisture, soil temperature, 
		Soil Moisture Depths: 0.07 - 0.07 m, 0.10 - 0.10 m, 0.20 - 0.20 m
		Soil Moisture Sensors: CS616, CS650, CS655, 

	MAQU
		Abstract: The MAQU network consist of 20 stations located at the eastern edge of the Tibetan Plateau in Central China at an altitude of more than 3000 m .a.s.l. The stations are operated by University of Twente, Faculty of Geo-Information Science and  Earth Observation (ITC), and the Chinese Academy of Science - Cold and Arid Regions Environmental and Engineering Research Institute (CAS-CAREERI). The soil moisture datasets were kindly provided by Laura Dente and Bob Su from the ITC and J. Wen of the CAS.
		Continent: Asia
		Country: China
		Stations: 27
		Status: running
		Data Range: from 2008-07-01 
		Type: project
		Url: https://www.itc.nl/about-itc/organization/scientific-departments/water-resources/earth-observation-sites/tibetan-plateau-peoples-republic-of-china/
		Reference: Dente, L., Su, Z. & Wen, J. (2012), ‘Validation of smos soil moisture products over the maqu and twente regions’, Sensors 12(8), 9965–9986, https://doi.org/10.3390/s120809965;

Su, Z., Wen, J., Dente, L., van der Velde, R., Wang, L., Ma, Y., Yang, K., and Hu, Z. 2011, The Tibetan Plateau observatory of plateau scale soil moisture and soil temperature (Tibet-Obs) for quantifying uncertainties in coarse resolution satellite and model products, Hydrol. Earth Syst. Sci., 15, 2303–2316, 2011, https://doi.org/10.5194/hess-15-2303-2011;

		Variables: soil temperature, soil moisture, 
		Soil Moisture Depths: 0.05 - 0.05 m, 0.10 - 0.10 m, 0.20 - 0.20 m, 0.40 - 0.40 m, 0.80 - 0.80 m
		Soil Moisture Sensors: ECH20 EC-TM, 

	METEROBS
		Abstract: 
		Continent: Europe
		Country: Italy
		Stations: 1
		Status: inactive
		Data Range: from 1986-01-01 
		Type: project
		Url: http://mistrals.sedoo.fr/HyMeX/Plateform-search?datsId=532
		Reference: We acknowledge the work of Nazzareno Diodato and the METEROBS team in support of the ISMN;
		Variables: soil moisture, 
		Soil Moisture Depths: 0.10 - 0.10 m, 0.20 - 0.20 m, 0.30 - 0.30 m, 0.40 - 0.40 m, 0.50 - 0.50 m
		Soil Moisture Sensors: EnviroSCAN, 

	MOL-RAO
		Abstract: This network is operated from the German Meteorological Service and consists of two stations. While the Station Falkenberg has a grass type vegetation, Kehrigk is situated in a pine forest. Volumetric soil moisture, soil temperature, air temperature and precipitation are provied for the period from 2003 to 2008. 
		Continent: Europe
		Country: Germany
		Stations: 2
		Status: running
		Data Range: from 2003-01-01 
		Type: project
		Url: http://www.dwd.de/mol
		Reference: Site and Data Report for the Lindenberg Reference Site in CEOP - Phase 1, Berichte des Deutschen Wetterdienstes, 230, Offenbach am Main;
		Variables: air temperature, precipitation, soil moisture, soil temperature, 
		Soil Moisture Depths: 0.08 - 0.08 m, 0.10 - 0.10 m, 0.15 - 0.15 m, 0.20 - 0.20 m, 0.30 - 0.30 m, 0.45 - 0.45 m, 0.60 - 0.60 m, 0.90 - 0.90 m, 1.50 - 1.50 m
		Soil Moisture Sensors: TRIME-EZ, 

	MONGOLIA
		Abstract: Soil moisture datasets for 44 stations were collected by the National Agency of Meterology, Hydrology and Environment Monitoring in Ulaanbaatar. All observations were taken using the gravimetric technique and initially provided as volumetric plant-available water [%]. Volumetric soil moisture [m^3/m^3] was calculated by first extracting texture properties of all sites from the Harmonized World Soil Database and subsequently calculating the wilting levels for all stations using the equations of Saxton and Rawls (2006). 
Soil moisture measurements are provided three times a month from 1964 to 2002 during the warm period of the year, which runs from April until the end of October.
		Continent: Asia
		Country: Mongolia
		Stations: 44
		Status: inactive
		Data Range: from 1964-04-08  to 2002-10-28
		Type: project
		Url: 
		Reference: Robock, Alan, Konstantin Y. Vinnikov, Govindarajalu Srinivasan, Jared K. Entin, Steven E. Hollinger, Nina A. Speranskaya, Suxia Liu, and A. Namkhai, 2000: The Global Soil Moisture Data Bank. Bull. Amer. Meteorol. Soc., 81, 1281-1299, https://doi.org/10.1175/1520-0477(2000)081<1281:TGSMDB>2.3.CO,2;
		Variables: soil moisture, 
		Soil Moisture Depths: 0.00 - 0.10 m, 0.10 - 0.20 m, 0.20 - 0.30 m, 0.30 - 0.40 m, 0.40 - 0.50 m, 0.50 - 0.60 m, 0.60 - 0.70 m, 0.70 - 0.80 m, 0.80 - 0.90 m, 0.90 - 1.00 m
		Soil Moisture Sensors: Coring device-auger, 

	MySMNet
		Abstract: 
		Continent: Asia
		Country: Malaysia
		Stations: 7
		Status: running
		Data Range: from 2014-06-01 
		Type: project
		Url: 
		Reference: Kang, C. S., Kanniah, K. D. & Kerr, Y. H. (2019), ‘Calibration of smos soil moisture retrieval algorithm: A case of tropical site in malaysia’, IEEE Transactions on Geoscience and Remote Sensing 57(6), 3827–3839, https://doi.org/10.1109/TGRS.2018.2888535;

		Variables: soil moisture, soil temperature, 
		Soil Moisture Depths: 0.00 - 0.05 m, 0.45 - 0.50 m, 0.95 - 1.00 m
		Soil Moisture Sensors: WaterScout SM100, 

	NAQU
		Abstract: 
		Continent: Asia
		Country: China
		Stations: 11
		Status: running
		Data Range: from 2006-07-01 
		Type: campaign
		Url: https://www.itc.nl/about-itc/organization/scientific-departments/water-resources/earth-observation-sites/tibetan-plateau-peoples-republic-of-china/
		Reference: Dente, L., Su, Z. & Wen, J. (2012), ‘Validation of smos soil moisture products over the maqu and twente regions’, Sensors 12(8), 9965–9986, https://doi.org/10.3390/s120809965;

Su, Z., Wen, J., Dente, L., van der Velde, R., Wang, L., Ma, Y., Yang, K., and Hu, Z. 2011, The Tibetan Plateau observatory of plateau scale soil moisture and soil temperature (Tibet-Obs) for quantifying uncertainties in coarse resolution satellite and model products, Hydrol. Earth Syst. Sci., 15, 2303–2316, 2011, https://doi.org/10.5194/hess-15-2303-2011;
		Variables: soil moisture, soil temperature, 
		Soil Moisture Depths: 0.05 - 0.05 m, 0.10 - 0.10 m, 0.20 - 0.20 m, 0.40 - 0.40 m, 0.80 - 0.80 m
		Soil Moisture Sensors: 5TM, 

	NGARI
		Abstract: 
		Continent: Asia
		Country: China
		Stations: 23
		Status: running
		Data Range: from 2010-07-01 
		Type: campaign
		Url: https://www.itc.nl/about-itc/organization/scientific-departments/water-resources/earth-observation-sites/tibetan-plateau-peoples-republic-of-china/
		Reference: Dente, L., Su, Z. & Wen, J. (2012), ‘Validation of smos soil moisture products over the maqu and twente regions’, Sensors 12(8), 9965–9986, https://doi.org/10.3390/s120809965;

Su, Z., Wen, J., Dente, L., van der Velde, R., Wang, L., Ma, Y., Yang, K., and Hu, Z. 2011, The Tibetan Plateau observatory of plateau scale soil moisture and soil temperature (Tibet-Obs) for quantifying uncertainties in coarse resolution satellite and model products, Hydrol. Earth Syst. Sci., 15, 2303–2316, 2011, https://doi.org/10.5194/hess-15-2303-2011;
		Variables: soil moisture, soil temperature, 
		Soil Moisture Depths: 0.05 - 0.05 m, 0.10 - 0.10 m, 0.20 - 0.20 m, 0.40 - 0.40 m, 0.80 - 0.80 m
		Soil Moisture Sensors: 5TM, 

	NVE
		Abstract: Ground water and soil moisture network

		Continent: Europe
		Country: Norway
		Stations: 6
		Status: running
		Data Range: from 1989-06-15 
		Type: permanent

		Url: https://www.nve.no/hydrology/?ref=mainmenu
		Reference: We acknowledge the work of Fred Wenger and the Norwegian water resources and energy direcorate (NVE) team in support of the ISMN;
		Variables: air temperature, precipitation, soil moisture, soil temperature, 
		Soil Moisture Depths: 0.10 - 0.10 m, 0.20 - 0.20 m, 0.30 - 0.30 m, 0.40 - 0.40 m, 0.60 - 0.60 m, 1.00 - 1.00 m
		Soil Moisture Sensors: Decagon 5 TM, Delta-T, Sutron, 

	ORACLE
		Abstract: 
		Continent: Europe
		Country: France
		Stations: 6
		Status: running
		Data Range: from 1985-10-18 
		Type: project
		Url: http://gisoracle.irstea.fr/?lang=en;https://bdoh.irstea.fr/ORACLE/
		Reference: We acknowledge the work of Arnaud Blanchouin and ORACLE team of the Institut national de recherche en sciences et technologies pour l"environment et l"agriculture, France in support of the ISMN;
		Variables: air temperature, precipitation, soil moisture, soil temperature, 
		Soil Moisture Depths: 0.03 - 0.03 m, 0.05 - 0.05 m, 0.06 - 0.06 m, 0.10 - 0.10 m, 0.15 - 0.15 m, 0.20 - 0.20 m, 0.25 - 0.25 m, 0.30 - 0.30 m, 0.35 - 0.35 m, 0.40 - 0.40 m, 0.45 - 0.45 m, 0.50 - 0.50 m, 0.55 - 0.55 m, 0.65 - 0.65 m, 0.70 - 0.70 m, 0.75 - 0.75 m, 0.85 - 0.85 m, 0.90 - 0.90 m, 0.95 - 0.95 m, 1.05 - 1.05 m, 1.10 - 1.10 m, 1.15 - 1.15 m, 1.30 - 1.30 m, 1.35 - 1.35 m, 1.50 - 1.50 m, 1.55 - 1.55 m
		Soil Moisture Sensors: Solo 40, ThetaProbe ML2X, TRASE 16, 

	OZNET
		Abstract: 
		Continent: Oceania
		Country: Australia
		Stations: 38
		Status: running
		Data Range: from 2001-01-01 
		Type: project
		Url: http://www.oznet.org.au/
		Reference: Smith, A. B., J. P.Walker, A. W.Western, R. I.Young, K. M.Ellett, R. C.Pipunic, R. B.Grayson, L.Siriwardena, F. H. S.Chiew, and H.Richter (2012), The Murrumbidgee soil moisture monitoring network data set, Water Resour. Res., 48, W07701, https://doi.org/10.1029/2012WR011976;

Young, R., Walker, J., Yeoh, N., Smith, A., Ellett, K., Merlin, O. and Western, A., 2008. Soil moisture and meteorological observations from the murrumbidgee catchment. Department of Civil and Environmental Engineering, The University of Melbourne;
		Variables: precipitation, soil temperature, soil moisture, soil suction, 
		Soil Moisture Depths: 0.00 - 0.05 m, 0.00 - 0.08 m, 0.00 - 0.30 m, 0.15 - 0.25 m, 0.30 - 0.60 m, 0.57 - 0.87 m, 0.60 - 0.90 m
		Soil Moisture Sensors: CS615, CS616, EnviroSCAN, Stevens Hydra Probe, 

	PBO_H2O
		Abstract: The soil moisture data is measured using GPS reflections.
		Continent: Americas
		Country: USA
		Stations: 163
		Status: inactive
		Data Range: from 2007-01-01 
		Type: project
		Url: https://gnss-reflections.org/maps?product=smc
		Reference: Kristine M. Larson, Eric E. Small, Ethan D. Gutmann, Andria L. Bilich, John J. Braun, Valery U. Zavorotny: Use of GPS receivers as a soil moisture network for water cycle studies. GEOPHYSICAL RESEARCH LETTERS, VOL. 35, L24405, https://doi.org/10.1029/2008GL036013, 2008;
		Variables: precipitation, air temperature, soil moisture, snow depth, 
		Soil Moisture Depths: 0.00 - 0.05 m
		Soil Moisture Sensors: GPS, 

	PTSMN
		Abstract: 
		Continent: Oceania
		Country: New Zealand
		Stations: 20
		Status: running
		Data Range: from 2016-11-01 
		Type: project
		Url: 
		Reference: Hajdu, I., Yule, I., Bretherton, M., Singh, R. & Hedley, C. (2019), ‘Field performance assessment and calibration of multi-depth aquacheck capacitance-based soil moisture probes under permanent pasture for hill country soils’, Agricultural Water Management 217, 332–345. https://doi.org/10.1016/J.AGWAT.2019.03.002;
		Variables: soil moisture, 
		Soil Moisture Depths: 0.07 - 0.13 m, 0.17 - 0.23 m, 0.27 - 0.33 m, 0.37 - 0.43 m
		Soil Moisture Sensors: AquaCheck, 

	REMEDHUS
		Abstract: 
		Continent: Europe
		Country: Spain
		Stations: 24
		Status: running
		Data Range: from 2005-01-01 
		Type: project
		Url: http://campus.usal.es/~hidrus/
		Reference: Gonzalez-Zamora, A., Sanchez, N., Pablos, M. & Martinez-Fernandez, J. (2018), ‘Cci soil moisture assessment with smos soil moisture and in situ data under different environmental conditions and spatial scales in spain’, Remote Sensing of Environment 225, https://doi.org/10.1016/j.rse.2018.02.010;
		Variables: soil temperature, soil moisture, 
		Soil Moisture Depths: 0.00 - 0.05 m
		Soil Moisture Sensors: Stevens Hydra Probe, 

	RISMA
		Abstract: In 2010 and 2011, Agriculture and Agri-Food Canada (AAFC), with the collaboration of Environment Canada, established three in situ monitoring networks near Kenaston (Saskatchewan), Carman (Manitoba) and Casselman (Ontario) as part of the Sustainable Agriculture Environmental Systems (SAGES) project titled Earth Observation Information on Crops and Soils for Agri-Environmental Monitoring in Canada.
		Continent: Americas
		Country: Canada
		Stations: 24
		Status: running
		Data Range: from 2013-06-15 
		Type: project
		Url: https://agriculture.canada.ca/en/agricultural-production/soil-and-land/real-time-situ-soil-monitoring-agriculture
		Reference: Ojo, E. R., Bullock, P. R., L’Heureux, J., Powers, J., McNairn, H., & Pacheco, A. (2015). Calibration and evaluation of a frequency domain reflectometry sensor for real-time soil moisture monitoring. Vadose Zone Journal, 14(3), https://doi.org/10.2136/vzj2014.08.0114;

L’Heureux, J. (2011). 2011 Installation Report for AAFC‐ SAGES Soil Moisture Stations in Kenaston, SK. Agriculture;

Canisius, F. (2011). Calibration of Casselman, Ontario Soil Moisture Monitoring Network, Agriculture and Agri‐Food Canada, Ottawa, ON, 37pp;
		Variables: precipitation, soil temperature, air temperature, soil moisture, 
		Soil Moisture Depths: 0.00 - 0.05 m, 0.05 - 0.05 m, 0.20 - 0.20 m, 0.50 - 0.50 m, 1.00 - 1.00 m, 1.50 - 1.50 m
		Soil Moisture Sensors: Hydraprobe II Sdi-12, 

	RSMN
		Abstract: The project proposal aims at paving the way for the utilisation of satellite derived soil moisture products in Romania, creating the framework for the validation and evaluation of actual & future satellite microwave soil moisture derived products, demonstrating its value, and by developing the necessary expertise for successfuly approaching implementations in the Societal Benefit Areas (as they were defined in GEOSS)

		Continent: Europe
		Country: Romania
		Stations: 20
		Status: running
		Data Range: from 2014-04-09 
		Type: meteo
		Url: http://assimo.meteoromania.ro
		Reference: We acknowledge the work of Andrei Diamandi and Adelina Mihai and the Romanian National Meteorological Administration team in support of the ISMN;
		Variables: air temperature, precipitation, soil moisture, soil temperature, 
		Soil Moisture Depths: 0.00 - 0.05 m
		Soil Moisture Sensors: 5TM, 

	RU-ADDFerti
		Abstract: 
		Continent: Europe
		Country: Germany
		Stations: 30
		Status: running
		Data Range: from 2023-06-01 
		Type: project
		Url: 
		Reference: We acknowledge the work of Alexander Steiger and the RU-ADDFerti network team in support of the ISMN;
		Variables: soil temperature, soil moisture, 
		Soil Moisture Depths: 0.25 - 0.25 m, 0.50 - 0.50 m
		Soil Moisture Sensors: LSE01, 

	Ru_CFR
		Abstract: 
		Continent: Europe
		Country: Russia
		Stations: 2
		Status: running
		Data Range: from 2015-05-25 
		Type: fluxnet
		Url: 
		Reference: We acknowledge the work of Andrej Varlagin and Adelina Mihai and the Ru_CFR team of the A.N. Severtsov Institute of Ecology and Evolution, Russian Academy of Sciences team in support of the ISMN;

		Variables: air temperature, precipitation, soil moisture, soil temperature, 
		Soil Moisture Depths: 0.05 - 0.05 m, 0.15 - 0.15 m, 0.50 - 0.50 m, 0.80 - 0.80 m, 1.00 - 1.00 m
		Soil Moisture Sensors: Hydraprobe II, 

	RUSWET-AGRO
		Abstract: Soil moisture datasets for 78 districts have been prepared by Vladimir Zabelin from the Russian Hydrometeorological Center, Moscow, Russia. The soil moisture observations were taken using the gravimetric technique and were provided as volumetric plant-available water [%] for the upper 20 cm and 1 m soil layers at agricultural fields. Thus volumetric soil moisture [m^3/m^3] was calculated by first extracting texture properties of all sites from the Harmonized World Soil Database and subsequently calculating the wilting levels for all stations using the equations of Saxton and Rawls (2006).
As this dataset was designed for monitoring of soil moisture at agricultural fields datasets were seperately available for spring cereal crops and winter cereal crops for all locations. In the ISMN stations with the suffix "a" represent spring cereal crops while stations with "b" illustrate winter cereal crops. Measurements are available during the growing period from 1987 to 1988, three times a month.
		Continent: Asia
		Country: Former Soviet Union
		Stations: 212
		Status: inactive
		Data Range: from 1986-12-28  to 1988-12-28
		Type: project
		Url: 
		Reference: Robock, Alan, Konstantin Y. Vinnikov, Govindarajalu Srinivasan, Jared K. Entin, Steven E. Hollinger, Nina A. Speranskaya, Suxia Liu, and A. Namkhai, 2000: The Global Soil Moisture Data Bank. Bull. Amer. Meteorol. Soc., 81, 1281-1299, https://doi.org/10.1175/1520-0477(2000)081<1281:TGSMDB>2.3.CO,2;
		Variables: soil moisture, 
		Soil Moisture Depths: 0.00 - 0.20 m, 0.00 - 1.00 m
		Soil Moisture Sensors: n.s., 

	RUSWET-GRASS
		Abstract: This network combines the two datasets RUSWET-GRASS-50STA and RUSWET-GRASS-130STA of the Global Soil Moisture Data Bank by Alan Robock and Konstantin Vinnikov. Soil moisture information at 122 meterological stations of the Former Soviet Union were collected. The results of soil moisture gravimetric measurements, which were taken at plots with natural grass type vegetation, were initially provided as volumetric plant-available water [%]. Volumetric soil moisture [m^3/m^3] was calculated by first extracting texture properties of all sites from the Harmonized World Soil Database and subsequently calculating the wilting levels for all stations using the equations of Saxton and Rawls (2006).
For the period 1952 to 1985 soil moisture measurements for the layer 0-100 cm are provided for some stations. Measurements of the upper 10 cm soil layer are available for the period 1977 to 1985.
		Continent: Asia
		Country: Former Soviet Union
		Stations: 122
		Status: inactive
		Data Range: from 1972-01-08  to 1985-12-28
		Type: project
		Url: 
		Reference: Robock, Alan, Konstantin Y. Vinnikov, Govindarajalu Srinivasan, Jared K. Entin, Steven E. Hollinger, Nina A. Speranskaya, Suxia Liu, and A. Namkhai, 2000: The Global Soil Moisture Data Bank. Bull. Amer. Meteorol. Soc., 81, 1281-1299, https://doi.org/10.1175/1520-0477(2000)081<1281:TGSMDB>2.3.CO,2;
		Variables: soil moisture, 
		Soil Moisture Depths: 0.00 - 0.10 m, 0.00 - 1.00 m
		Soil Moisture Sensors: n.s., 

	RUSWET-VALDAI
		Abstract: Soil moisture datasets for 3 catchments located at the Valdai basin were collected by the State Hydrological Institute in St. Petersburg from 1960 to 1990. The catchments differ in their vegetation, Usadievskiy has mostly grassland vegetation, Tayozhniy a mature forest and Sinaya Gnilka was articifally forested in 1961. 
The soil moisture was computed by using data from 9-11 observational points distributed over the basin area. The resulting files contain monthly means of soil moisture for the layers of 0-20, 0-50 and 0-100 cm. These files were downloaded from the "Global Soil Moisture Database" and transformed into volumetric soil moisture [m^3/m^3]. Thus soil moisture values are available as monthly means, precipitation as monthly totals [mm] and temperature [°C] as daily values. Other variables measured but not included into the ISMN are runoff, averaged water table depth, averaged water equivalent snow depth and averaged evaporation.
		Continent: Asia
		Country: Former Soviet Union
		Stations: 3
		Status: inactive
		Data Range: from 1960-01-15  to 1990-12-15
		Type: project
		Url: 
		Reference: Robock, Alan, Konstantin Y. Vinnikov, Govindarajalu Srinivasan, Jared K. Entin, Steven E. Hollinger, Nina A. Speranskaya, Suxia Liu, and A. Namkhai, 2000: The Global Soil Moisture Data Bank. Bull. Amer. Meteorol. Soc., 81, 1281-1299, https://doi.org/10.1175/1520-0477(2000)081<1281:TGSMDB>2.3.CO,2;
		Variables: air temperature, precipitation, soil moisture, soil temperature, 
		Soil Moisture Depths: 0.00 - 0.20 m, 0.00 - 0.50 m, 0.00 - 1.00 m
		Soil Moisture Sensors: n.s., 

	SASMAS
		Abstract: 
		Continent: Oceania
		Country: Australia
		Stations: 14
		Status: running
		Data Range: from 2002-01-01 
		Type: project
		Url: http://www.eng.newcastle.edu.au/sasmas/SASMAS/sasdata.html
		Reference: Rüdiger, C, Hancock, G. R., Hemakumara, H. M., Jacobs, B., Kalma, J. D.,
Martinez, C., Thyer, M., Walker, J. P., Wells, T. and Willgoose, G. R.,
2007. The Goulburn River Experimental Catchment Data Set. Water
Resources Research, 43, W10403, https://doi.org/10.1029/2006WR005837;
		Variables: soil moisture, soil temperature, 
		Soil Moisture Depths: 0.00 - 0.05 m, 0.00 - 0.30 m
		Soil Moisture Sensors: CS616, Stevens Hydra Probe, 

	SCAN
		Abstract: Soil Climate Analysis Network contains 239 stations all over the USA including stations in Alaska, Hawaii, Puerto Rico or even one in Antarctica. Apart from soil moisture and soil temperature, also precipitation and air temperature are measured. Some stations have also additional measurements of snow depth and snow water equivalent. Almost 150 stations are updated on daily basis. The network is operated by the USDA NRCS National Water and Climate Center with assistance from the USDA NRCS National Soil Survey Center.
		Continent: Americas
		Country: USA
		Stations: 239
		Status: running
Data Range: 

		Type: project
		Url: http://www.wcc.nrcs.usda.gov/
		Reference: Schaefer, G., Cosh, M. & Jackson, T. (2007), ‘The usda natural resources conservation service soil climate analysis network (scan)’, Journal of Atmospheric and Oceanic Technology - J ATMOS OCEAN TECHNOL 24, https://doi.org/10.1175/2007JTECHA930.1;
		Variables: snow water equivalent, precipitation, soil temperature, air temperature, soil moisture, snow depth, 
		Soil Moisture Depths: 0.03 - 0.03 m, 0.05 - 0.05 m, 0.10 - 0.10 m, 0.15 - 0.15 m, 0.20 - 0.20 m, 0.25 - 0.25 m, 0.28 - 0.28 m, 0.30 - 0.30 m, 0.38 - 0.38 m, 0.41 - 0.41 m, 0.46 - 0.46 m, 0.51 - 0.51 m, 0.61 - 0.61 m, 0.66 - 0.66 m, 0.69 - 0.69 m, 0.76 - 0.76 m, 0.84 - 0.84 m, 0.89 - 0.89 m, 1.02 - 1.02 m, 1.09 - 1.09 m, 1.14 - 1.14 m, 1.30 - 1.30 m, 1.40 - 1.40 m, 1.52 - 1.52 m, 2.03 - 2.03 m
		Soil Moisture Sensors: Hydraprobe Digital Sdi-12 (2.5 Volt), n.s., Hydraprobe Analog (2.5 Volt), Hydraprobe Analog (5.0 Volt), Hydraprobe Digital Sdi-12 Thermistor (linear), 

	SD_DEM
		Abstract: Meteorological data and soil data have been collected at a site in the central Sudan from 2002 to 2012. The site is a sparse savanna in the semiarid region of Sudan. In addition to basic meteorological variables, soil properties (temperature, water content, and heat flux) and radiation (global radiation, net radiation, and photosynthetic active radiation) were measured. The dataset has a temporal resolution of 30 minutes and provides general data for calibration and validation of ecosystem models and remote-sensing-based assessments, and it is relevant for studies of ecosystem properties and processes.
		Continent: Africa
		Country: Sudan
		Stations: 1
		Status: running
		Data Range: from 2002-02-08 
		Type: project
		Url: http://dx.doi.org/10.7167/2013/297973
		Reference: Jonas Ardö, A 10-Year Dataset of Basic Meteorology and Soil Properties in Central Sudan, Dataset Papers in Geosciences, vol. 2013, Article ID 297973, 6 pages, 2013, https://doi.org/10.7167/2013/297973/dataset;
		Variables: air temperature, precipitation, soil moisture, soil temperature, 
		Soil Moisture Depths: 0.05 - 0.05 m, 0.10 - 0.10 m, 0.15 - 0.15 m, 0.30 - 0.30 m, 0.60 - 0.60 m, 1.00 - 1.00 m, 1.50 - 1.50 m, 2.00 - 2.00 m
		Soil Moisture Sensors: CS615, CS616, HYDRA, 

	SKKU
		Abstract: 
		Continent: Asia
		Country: Korea
		Stations: 15
		Status: running
Data Range: 

		Type: project
		Url: 
		Reference: Nguyen, H. H., Kim, H. & Choi, M. (2017), ‘Evaluation of the soil water content using cosmic-ray neutron probe in a heterogeneous monsoon climate-dominated region’, Advances in Water Resources 108, 125–138., https://doi.org/10.1016/j.advwatres.2017.07.020;
		Variables: soil moisture, soil temperature, 
		Soil Moisture Depths: 0.05 - 0.05 m, 0.10 - 0.10 m, 0.20 - 0.20 m, 0.30 - 0.30 m, 0.40 - 0.40 m
		Soil Moisture Sensors: 5TM, 

	SMN-SDR
		Abstract: The SMN-SDR was established during the Soil Moisture Experiment in the Luan River of China (SMELR, Zhao et al., 2020), from July 18, 2018 to September 28, 2018.
The coverage of entire network is about 10,000 km2 (115.5-116.5°E, 41.5-42.5°N). 
The topography of the SMN-SDR is relatively flat, and land surfaces are typically dominated by grasslands and croplands.
A total of 34 stations are set up in the network with three sampling scales including 100 km (large scale), 50 km (medium scale), and 10 km (small scale).
These were initially designed to match different scales of land surface modelling and various satellite products (SMAP, SMOS, AMSR2, and downscaled soil moisture products).
The soil moisture sensors used are Decagon 5TM with five measuring depths (3, 5, 10, 20, and 50 cm) installed for each station. 
Of the 34 station, there are 20 stations equipped with HOBO rain gauges.
Undisturbed soil samples at each layer of soil for each station was taken to analyze the gravimetric/volumetric water content, bulk density, and soil texture for a further specific calibration. 
The power supply is provided by solar panels and all data can be transmitted wirelessly to a server. 
The sampling interval of the data recording time is 10 (before June 2019) or 15 minutes (after June 2019).
This network can improve the comprehensive observation capabilities of key water cycle parameters in the ShanDian River basin and provide long-term ground reference data for satellite- and model-based soil moisture products.
		Continent: Asia
		Country: China
		Stations: 34
		Status: running
		Data Range: from 2018-07-18 
		Type: Soil moisture experiment in the Luan River
		Url: https://doi.org/10.11888/Soil.tpdc.271425

https://doi.org/10.11888/Soil.tpdc.271434
		Reference: Zhao, T., Shi, J., Lv, L., Xu, H., Chen, D., Cui, Q., Jackson, T.J., Yan, G., Jia, L., Chen, L., Zhao, K., Zheng, X., Zhao, L., Zheng, C., Ji, D., Xiong, C., Wang, T., Li, R., Pan, J., Wen, J., Yu, C., Zheng, Y., Jiang, L., Chai, L., Lu, H., Yao, P., Ma, J., Lv, H., Wu, J., Zhao, W., Yang, N., Guo, P., Li, Y., Hu, L., Geng, D., Zhang, Z., 2020. Soil moisture experiment in the Luan River supporting new satellite mission opportunities. Remote Sensing of Environment 240, 111680. https://doi.org/10.1016/j.rse.2020.111680;

Zheng, J., Zhao, T., Lü, H., Shi, J., Cosh, M.H., Ji, D., Jiang, L., Cui, Q., Lu, H., Yang, K., Wigneron, J.-P., Li, X., Zhu, Y., Hu, L., Peng, Z., Zeng, Y., Wang, X., Kang, C.S., 2022. Assessment of 24 soil moisture datasets using a new in situ network in the Shandian River Basin of China. Remote Sensing of Environment 271, 112891. https://doi.org/10.1016/j.rse.2022.112891;
		Variables: precipitation, soil temperature, soil moisture, 
		Soil Moisture Depths: 0.03 - 0.03 m, 0.05 - 0.05 m, 0.10 - 0.10 m, 0.20 - 0.20 m, 0.50 - 0.50 m
		Soil Moisture Sensors: 5TM, 

	SMOSMANIA
		Abstract: 
		Continent: Europe
		Country: France
		Stations: 22
		Status: running
		Data Range: from 2003-01-01 
		Type: project
		Url: http://www.hymex.org
		Reference: Calvet, J.-C., Fritz, N., Berne, C., Piguet, B., Maurel, W. & Meurey, C. (2016), ‘Deriving pedotransfer functions for soil quartz fraction in southern france from reverse modeling’, SOIL 2(4), 615–629, https://doi.org/10.5194/soil-2-615-2016;

Albergel, C., Rüdiger, C., Pellarin, T., Calvet, J.-C., Fritz, N., Froissard, F., Suquia, D., Petitpa, A., Piguet, B., and Martin, E.: From near-surface to
root-zone soil moisture using an exponential filter: an assessment of the method based
on insitu observations and model simulations, Hydrol. Earth Syst. Sci., 12, 1323–1337, 2008, https://doi.org/10.5194/hess-12-1323-2008;

Calvet, J.-C., Fritz, N., Froissard, F., Suquia, D., Petitpa, A., and Piguet, B.: In situ soil moisture observations for the CAL/VAL of SMOS: the SMOSMANIA network, International Geoscience and Remote Sensing Symposium, IGARSS, Barcelona, Spain, 23-28 July 2007, 1196-1199, https://doi.org/10.1109/IGARSS.2007.4423019, 2007.;
		Variables: soil moisture, soil temperature, 
		Soil Moisture Depths: 0.05 - 0.05 m, 0.10 - 0.10 m, 0.20 - 0.20 m, 0.30 - 0.30 m
		Soil Moisture Sensors: ThetaProbe ML2X, ThetaProbe ML3, 

	SNOTEL
		Abstract: The Natural Resources Conservation Service (NRCS) installs, operates, and maintains an extensive, automated system to collect snowpack and related climatic data in the Western United States called SNOTEL (for SNOwpack TELemetry). The system evolved from NRCS"s Congressional mandate in the mid-1930"s "to measure snowpack in the mountains of the West and forecast the water supply." The programs began with manual measurements of snow courses; since 1980, SNOTEL has reliably and efficiently collected the data needed to produce water supply forecasts and to support the resource management activities of NRCS and others.
		Continent: Americas
		Country: USA
		Stations: 460
		Status: running
		Data Range: from 1980-01-01 
		Type: project
		Url: http://www.wcc.nrcs.usda.gov/
		Reference: Leavesley et al (2010), ‘A modelling framework for improved agricultural water-supply forecasting’;

Leavesley, G., David, O., Garen, D., Lea, J., Marron, J., Pagano, T., Perkins, T. & Strobel, M. (2008), ‘A modeling framework for improved agricultural water supply forecasting’, AGU Fall Meeting Abstracts;
		Variables: snow water equivalent, precipitation, soil temperature, air temperature, soil moisture, snow depth, 
		Soil Moisture Depths: 0.00 - 0.00 m, 0.05 - 0.05 m, 0.08 - 0.08 m, 0.10 - 0.10 m, 0.20 - 0.20 m, 0.28 - 0.28 m, 0.30 - 0.30 m, 0.36 - 0.36 m, 0.41 - 0.41 m, 0.46 - 0.46 m, 0.51 - 0.51 m, 0.61 - 0.61 m, 0.81 - 0.81 m, 0.94 - 0.94 m, 0.99 - 0.99 m, 1.02 - 1.02 m
		Soil Moisture Sensors: Hydraprobe Analog (5.0 Volt), n.s., Hydraprobe Analog (2.5 Volt), Hydraprobe Digital Sdi-12 (2.5 Volt), 

	SOILSCAPE
		Abstract: 
		Continent: Americas
		Country: USA
		Stations: 224
		Status: running
Data Range: 

		Type: project
		Url: http://soilscape.usc.edu/
		Reference: Moghaddam, M., A.R. Silva, D. Clewley, R. Akbar, S.A. Hussaini, J. Whitcomb, R. Devarakonda, R. Shrestha, R.B. Cook, G. Prakash, S.K. Santhana Vannan, and A.G. Boyer. 2016. Soil Moisture Profiles and Temperature Data from SoilSCAPE Sites, USA. ORNL DAAC, Oak Ridge, Tennessee, USA. http://dx.doi.org/10.3334/ORNLDAAC/1339;

Moghaddam, M., D. Entekhabi, Y. Goykhman, K. Li, M. Liu, A. Mahajan, A. Nayyar, D. Shuman, and D. Teneketzis, A wireless soil moisture smart sensor web using physics-based optimal control: concept and initial demonstration IEEE-JSTARS, , vol. 3, no. 4, pp. 522-535, December 2010, https://doi.org/10.1109/JSTARS.2010.2052918;

Shuman, D. I., Nayyar, A., Mahajan, A., Goykhman, Y., Li, K., Liu, M., Teneketzis, D., Moghaddam, M. & Entekhabi, D. (2010), ‘Measurement scheduling for soil moisture sensing: From physical models to optimal control’, Proceedings of the IEEE 98(11), 1918–1933, https://doi.org/10.1109/JPROC.2010.2052532;
		Variables: precipitation, soil temperature, air temperature, soil moisture, 
		Soil Moisture Depths: 0.04 - 0.04 m, 0.05 - 0.05 m, 0.10 - 0.10 m, 0.13 - 0.13 m, 0.15 - 0.15 m, 0.17 - 0.17 m, 0.20 - 0.20 m, 0.23 - 0.23 m, 0.28 - 0.28 m, 0.29 - 0.29 m, 0.30 - 0.30 m, 0.32 - 0.32 m, 0.35 - 0.35 m, 0.36 - 0.36 m, 0.37 - 0.37 m, 0.38 - 0.38 m, 0.39 - 0.39 m, 0.40 - 0.40 m, 0.41 - 0.41 m, 0.42 - 0.42 m, 0.43 - 0.43 m, 0.45 - 0.45 m, 0.46 - 0.46 m, 0.47 - 0.47 m, 0.48 - 0.48 m, 0.50 - 0.50 m, 0.60 - 0.60 m, 0.70 - 0.70 m, 0.75 - 0.75 m, 0.90 - 0.90 m
		Soil Moisture Sensors: TEROS12, EC5, 5TM, 

	SONTE-China
		Abstract: An automatic soil observation network in typical ecosystems in China (SONTE-China) was established by the Aerospace Information Research Institute (AIR), Chinese Academy of Sciences (CAS), and National Engineering Research Center of Satellite Remote Sensing Applications(NECRSA), which were of the same unified sensor, unified sample design, unified acquisition depth, unified calibration method, unified data processing process. SONTE-China covers the main ecosystems (including grassland, farmland, desert and forest) and dry and wet climate zones in China. It can automatically obtain pixel scale soil temperature and soil moisture, which can not only serve the development and verification of high-resolution satellite remote sensing soil moisture products but also provide basic data for research in the fields of weather forecasting, flood forecasting, agricultural drought monitoring and water resources management.
		Continent: Asia
		Country: China
		Stations: 20
		Status: running
		Data Range: from 2019-01-01 
		Type: project
		Url: 
		Reference: Bingze Li, Chunmei Wang, Xingfa Gu, Xiang Zhou et al. 2022. Accuracy calibration and evaluation of capacitance-based soil moisture sensors for a variety of soil properties. Agricultural Water Management, 273, 107913. https://doi.org/10.1016/j.agwat.2022.107913.
Leran Han, Chunmei Wang, Tao Yu, Xingfa Gu et al. High-Precision soil moisture mapping based on multi-model coupling and background knowledge, over vegetated areas using Chinese GF-3 and GF-1 satellite data. Remote Sensing, 2020, 12(13):2123.
		Variables: soil moisture, soil temperature, 
		Soil Moisture Depths: 0.05 - 0.05 m, 0.10 - 0.10 m, 0.20 - 0.20 m, 0.40 - 0.40 m
		Soil Moisture Sensors: 5TM, 

	STEMS
		Abstract: Soil Moisture Network installed in rainfed vineyard plots, with different inter-rows soil management. The plots are also monitored for runoff and soil erosion. Weather data available from a station near the plots.
		Continent: Europe
		Country: Italy
		Stations: 4
		Status: running
		Data Range: from 2015-12-04 
		Type: campaign
		Url: https://sustag.to.cnr.it/index.php/cannona-db
		Reference: Darouich, H., Ramos, T.B., Pereira, L.S., Rabino, D., Bagagiolo, G., Capello, G., Simionesei, L., Cavallo, E., Biddoccu, M. Water Use and Soil Water Balance of Mediterranean Vineyards under Rainfed and Drip Irrigation Management: Evapotranspiration Partition and Soil Management Modelling for Resource Conservation. Water 2022, 14, 554. https://doi.org/10.3390/w14040554;

Capello G, Biddoccu M, Ferraris S, Cavallo E, 2019. Effects of tractor passes on hydrological and soil erosion processes in tilled and grassed vineyards. Water 2019, 11(10), 2118, https://doi.org/10.3390/w11102118;
		Variables: air temperature, precipitation, soil moisture, soil temperature, 
		Soil Moisture Depths: 0.10 - 0.10 m, 0.20 - 0.20 m, 0.30 - 0.30 m, 0.40 - 0.40 m, 0.50 - 0.50 m
		Soil Moisture Sensors: 5TM, EC5, 

	SWEX_POLAND
		Abstract: 
		Continent: Europe
		Country: Poland
		Stations: 6
		Status: inactive
		Data Range: from 2006-08-03  to 2013-05-06
		Type: project
		Url: 
		Reference: W. Marczewski, J. Slominski, E. Slominska, B. Usowicz, J. Usowicz, S. Romanov, O. Maryskevych, J. Nastula, and J. Zawadzki, 2010: Strategies for validating and directions for employing SMOS data, in the Cal-Val project SWEX (3275) for wetlands, Hydrol. Earth Syst. Sci. Discuss., 7, 7007–7057, https://doi.org/10.5194/hessd-7-7007-2010;
		Variables: precipitation, soil moisture, soil temperature, 
		Soil Moisture Depths: 0.00 - 0.02 m, 0.05 - 0.05 m, 0.10 - 0.10 m, 0.20 - 0.20 m, 0.30 - 0.30 m, 0.40 - 0.40 m, 0.50 - 0.50 m, 0.60 - 0.60 m, 0.60 - 1.00 m, 1.00 - 1.00 m
		Soil Moisture Sensors: D-LOG-mpts, PR2 - Profile Probe, 

	SW-WHU
		Abstract: Based on sensor web principle, this network provides continuous and high density monitoring data for soil moisture, soil temperature and meteorological conditions.
		Continent: Asia
		Country: China
		Stations: 7
		Status: running
		Data Range: from 2014-01-11 
		Type: project
		Url: http://202.114.118.60:9002/SensorWebPro/index.html
		Reference: Chen, N., Zhang,X., Wang, C.,2015. Integrated Open Geospatial Web Service enabled Cyber-physical Information Infrastructure for Precision Agriculture Monitoring. Computers and Electronics in Agriculture.111,78–91, https://doi.org/10.1016/j.compag.2014.12.009;

Chen, N., Xiao, C., Pu, F., Wang, Z., Wang, C., Gong, J. 2015. Cyber-Physical Geographical Information Service-Enabled Control of Diverse In-situ Sensors. Sensors. 15(2), 2565-2592, https://doi.org/10.3390/s150202565;

Zhang, X., Chen, N., Chen, Z., Wu, L., Li, X., Zhang, L., Di, L., Gong, J. & Li, D. (2018), ‘Geospatial sensor web: A cyber-physical infrastructure for geoscience research and application’, Earth-science reviews 185, 684–703, https://doi.org/10.1016/j.earscirev.2018.07.006;
		Variables: air temperature, precipitation, soil moisture, soil temperature, 
		Soil Moisture Depths: 0.10 - 0.10 m
		Soil Moisture Sensors: FY-H2, LVDSC12, SoilMoisture-AverageSensor, 

	TAHMO
		Abstract: The Trans-African HydroMeteorological Observatory (TAHMO) aims to develop a vast network of weather stations across Africa. Current and historic weather data is important for agricultural, climate monitoring, and many hydro-meteorological applications.
		Continent: Africa
		Country: Côte d'Ivoire, Nigeria, Ghana, Uganda, Rwanda, Kenya
		Stations: 70
		Status: running
		Data Range: from 2015-06-17 
		Type: project
		Url: https://tahmo.org/
		Reference: We acknowledge the work of Frank Annor and Nicolaas Cornelis van de Giesen and the Trans-African Hydro-Meterological Observatory (TAHMO) network community in support of the ISMN;
		Variables: precipitation, soil temperature, air temperature, soil moisture, 
		Soil Moisture Depths: 0.05 - 0.05 m, 0.10 - 0.10 m, 0.20 - 0.20 m, 0.30 - 0.30 m, 0.60 - 0.60 m, 2.00 - 2.00 m
		Soil Moisture Sensors: TEROS12, TEROS10, GS1, 

	TERENO
		Abstract: Soil moisture network in Germany, There are 4 observatories: in Northeastern Germany- Lowlan Observatory coordinated by German Research Centre of Geosciences, in Harz/Central Germany-  Lowland Observatory coordinated by Helmholtz Centre for Environmental Research, in Eifel/Lower Rhine Valley- Observatory coordinated by Research Centre Juelich and in Bavarian Alps/pre-Alps- Obervatory coordinated by Karlsruhe Institute of Technology and German Center for Environmental Health
		Continent: Europe
		Country: Germany
		Stations: 5
		Status: running
Data Range: 

		Type: meteo
		Url: https://www.tereno.net/joomla/index.php/overview
		Reference: Bogena, H.R., Montzka, C., Huisman, J.A., Graf, A., Schmidt, M., Stockinger, M., Von Hebel, C., Hendricks-Franssen, H.J., Van der Kruk, J., Tappe, W. and Lücke, A., 2018. The TERENO‐Rur hydrological observatory: A multiscale multi‐compartment research platform for the advancement of hydrological science. Vadose Zone Journal, 17(1), pp.1-22, https://doi.org/10.2136/vzj2018.03.0055;

Bogena, H. R. (2016), ‘Tereno: German network of terrestrial environmental observatories’, Journal of large-scale research facilities JLSRF 2, 52, http://dx.doi.org/10.17815/jlsrf-2-98;

Bogena, H., Kunkel, R., Puetz, T., Vereecken, H., Krueger, E., Zacharias, S., Dietrich, P., Wollschlaeger, U., Kunstmann, H., Papen, H. and Schmid, H.P., 2012. Tereno-long-term monitoring network for terrestrial environmental research. Hydrologie und Wasserbewirtschaftung, 56(3), pp.138-143, https://doi.org/DOI:%10.5675;

Zacharias, S., H.R. Bogena, L. Samaniego, M. Mauder, R. Fuß, T. Pütz, M. Frenzel, M. Schwank, C. Baessler, K. Butterbach-Bahl, O. Bens, E. Borg, A. Brauer, P. Dietrich, I. Hajnsek, G. Helle, R. Kiese, H. Kunstmann, S. Klotz, J.C. Munch, H. Papen, E. Priesack, H. P. Schmid, R. Steinbrecher, U. Rosenbaum, G. Teutsch, H. Vereecken. 2011. A Network of Terrestrial Environmental Observatories in Germany. Vadose Zone J. 10. 955–973, https://doi.org/10.2136/vzj2010.0139;
		Variables: air temperature, precipitation, soil moisture, soil temperature, 
		Soil Moisture Depths: 0.05 - 0.05 m, 0.20 - 0.20 m, 0.50 - 0.50 m
		Soil Moisture Sensors: Hydraprobe II Sdi-12, 

	TWENTE
		Abstract: The Twente region in the east of the Netherlands hold a network that monitors since 2009 profile soil moisture and temperature at twenty locations mostly in the rural environment. In addition, intensive measurement campaigns have been performed in 2009, 2015, 2016, and 2017 at number of measurement locations as a part of MSc and PhD research. The development of the network infrastructure has been supported through various EU and NWO-funded projects, and forthcoming data has been successfully used for the validation of satellite derived surface soil moisture products. The early development of the network has been described in Dente et al. (2011), but over the years the design of the network has altered considerably with the most recent updates documented in Van der Velde et al. (2021). In 2019, the Twente network has been operational for more than a decade and with this deposit we would like to make this data freely and well documented available for use by the science and professional communities. An accompanying data paper is under development and will be submitted to Earth System Science Data journal, which provide additional background information.
		Continent: Europe
		Country: Netherlands
		Stations: 44
		Status: running
		Data Range: from 2009-01-01 
		Type: project
		Url: https://doi.org/10.5194/essd-15-1889-2023
		Reference: Van der Velde, R., Benninga, H. J. F., Retsios, B., Vermunt, P. C., & Salama, M. S. (2023). Twelve years of profile soil moisture and temperature measurements in Twente, the Netherlands. Earth System Science Data, 15(4), 1889-1910.
https://doi.org/10.5194/essd-15-1889-2023
https://essd.copernicus.org/articles/15/1889/2023/

		Variables: soil temperature, soil moisture, 
		Soil Moisture Depths: 0.05 - 0.05 m, 0.10 - 0.10 m, 0.20 - 0.20 m, 0.40 - 0.40 m, 0.80 - 0.80 m
		Soil Moisture Sensors: 5TM, EC-TM, 

	TxSON
		Abstract: 
		Continent: Americas
		Country: USA
		Stations: 41
		Status: running
		Data Range: from 2014-10-01 
		Type: project
		Url: http://www.beg.utexas.edu/research/programs/txson
		Reference: Caldwell, T. G., T. Bongiovanni, M. H. Cosh, T. J. Jackson, A. Colliander, C. J. Abolt, R. Casteel, T. Larson, B. R. Scanlon, and M. H. Young (2019), The Texas Soil Observation Network: A comprehensive soil moisture dataset for remote sensing and land surface model validation, Vadose Zone Journal, 18:100034, doi:10.2136/vzj2019.04.0034
		Variables: soil temperature, soil moisture, 
		Soil Moisture Depths: 0.05 - 0.05 m
		Soil Moisture Sensors: CS655, 

	UDC_SMOS
		Abstract: Soil moisture datasets of 11 stations near the city Munich in Germany are provided.  The measurements are taken at five different layers up to a depth of 40 cm. At some stations soil moisture of a specific layer is measured by multiple sensors. All stations of this network are situated on grassland. Meteorological data for the stations 5, 14, 16, 49, 74, 80, 115, 118 and 125 is available for free via the website of the Bavarian State Research Center for Agriculture (http://www.wetter-by.de/).
The soil moisture network is run in cooperation with the Bavarian State Research Center for Agriculture. This work is carried out as part of the project SMOSHYD (FKZ 50EE0731) funded by the German Aerospace Centre (DLR).
		Continent: Europe
		Country: Germany
		Stations: 11
		Status: inactive
		Data Range: from 2007-11-08  to 2011-10-21
		Type: project
		Url: http://www.geographie.uni-muenchen.de/department/fiona/forschung/projekte/index.php?projekt_id=103
		Reference: Schlenz, F., Dall’Amico, J., Loew, A., Mauser, W. (2012): Uncertainty Assessment of the SMOS Validation in the Upper Danube Catchment. IEEE Transactions on Geoscience and Remote Sensing, 50(5), pp.1517–1529, https://doi.org/10.1109/TGRS.2011.2171694;

A. Loew, J. T. Dall"Amico, F. Schlenz, W. Mauser (2009): The Upper Danube soil moisture validation site: measurements and activities, paper presented at Earth Observation and Water Cycle conference, Frascati (Rome), 18 - 20 November 2009, to be published in ESA Special dataation SP-674, https://ui.adsabs.harvard.edu/abs/2009ESASP.674E..56L;
		Variables: soil moisture, 
		Soil Moisture Depths: 0.00 - 0.10 m, 0.05 - 0.05 m, 0.10 - 0.10 m, 0.20 - 0.20 m, 0.40 - 0.40 m
		Soil Moisture Sensors: EC5, EC5 I, EC5 II, EC5 III, EC5 IV, EC5 V, EC5 VI, EC-ET, EC-ET 2, IMKO TDR, IMKO TDR 1, IMKO TDR 2, 

	UMBRIA
		Abstract: The soil moisture network of the Civil Protection Functional Centre (CFD Umbria), together with the Research Institute for Geo-Hydrological Protection and Consiglio Nazionale delle Ricerca (abbreviated CNR-IRPI)  is located near the city of Perugia at Lake Trasimeno in the middle of Italy and consists of 7 stations. Four of these seven stations come from the old Network CRN_IRPI which are now part of the UMBRIA Network. One station became operational in the year 2002 and is providing soil moisture measurements at 3 layers and additionaly precipitation and air temperature observations. Other 3 stations, became operational in the year 2007 collect soil moisture measurements at 3 depths. The observed variables are sampled each hour.
		Continent: Europe
		Country: Italy
		Stations: 13
		Status: running
		Data Range: from 2002-10-09 
		Type: project
		Url: http://www.cfumbria.it/; http://hydrology.irpi.cnr.it/
		Reference: Brocca, L., Hasenauer, S., Lacava, T., Melone, F., Moramarco, T., Wagner, W., Dorigo, W., Matgen, P., Martínez-Fernández, J., Llorens, P., Latron, J., Martin, C., Bittelli, M. (2011). Soil moisture estimation through ASCAT and AMSR-E sensors: an intercomparison and validation study across Europe. Remote Sensing of Environment, 115, 3390-3408, https://doi.org/10.1016/j.rse.2011.08.003;

Brocca, L., Melone, F., Moramarco, T., Morbidelli, R. (2009). Antecedent wetness conditions based on ERS scatterometer data. Journal of Hydrology, 364 (1-2), 73-87, https://doi.org/10.1016/j.jhydrol.2008.10.007;

Brocca, L., Melone, F., Moramarco, T. (2008). On the estimation of antecedent wetness condition in rainfall-runoff modelling. Hydrological Processes, 22 (5), 629-642, https://doi.org/10.1002/hyp.6629;
		Variables: air temperature, precipitation, soil moisture, 
		Soil Moisture Depths: 0.05 - 0.15 m, 0.15 - 0.25 m, 0.25 - 0.35 m, 0.35 - 0.45 m, 0.45 - 0.55 m
		Soil Moisture Sensors: EnviroSCAN, EnviroSMART, ThetaProbe ML2X, 

	UMSUOL
		Abstract: 
		Continent: Europe
		Country: Italy
		Stations: 1
		Status: running
		Data Range: from 2004-08-31 
		Type: project
		Url: http://www.arpa.emr.it/sim/
		Reference: We acknowledge the work of Andrea Pasquali and the UMSUOL network team provided by the Agenzia Regionale Prevenzione Ambiente - Servizio Idro-Meteo-Clima (ARPA - SIMC) in support of the ISMN;

		Variables: soil moisture, 
		Soil Moisture Depths: 0.10 - 0.10 m, 0.25 - 0.25 m, 0.45 - 0.45 m, 0.70 - 0.70 m, 1.00 - 1.00 m, 1.35 - 1.35 m, 1.80 - 1.80 m
		Soil Moisture Sensors: TDR 100, 

	USCRN
		Abstract: Soil moisture NRT network USCRN (Climate Reference Network) in United States;the  datasets of 114 stations were collected and processed by the National Oceanicand Atmospheric Administration"s National Climatic Data Center (NOAA"s NCDC)
		Continent: Americas
		Country: USA
		Stations: 115
		Status: running
		Data Range: from 2009-06-09 
		Type: meteo
		Url: https://www.ncei.noaa.gov/access/crn/
		Reference: Bell, J. E., M. A. Palecki, C. B. Baker, W. G. Collins, J. H. Lawrimore, R. D. Leeper, M. E. Hall, J. Kochendorfer, T. P. Meyers, T. Wilson, and H. J. Diamond. 2013: U.S. Climate Reference Network soil moisture and temperature observations. J. Hydrometeorol., 14, 977-988, https://doi.org/10.1175/JHM-D-12-0146.1;
		Variables: surface temperature, precipitation, soil temperature, air temperature, soil moisture, 
		Soil Moisture Depths: 0.05 - 0.05 m, 0.10 - 0.10 m, 0.20 - 0.20 m, 0.50 - 0.50 m, 1.00 - 1.00 m
		Soil Moisture Sensors: Stevens Hydraprobe II Sdi-12, 

	USDA-ARS
		Abstract: The USDA-ARS watershed network was initiated as part of an EOS Aqua AMSR-E project in 2002. It consists of four sub-networks - Little River, Little Washita, Walnut Gulch and Reynolds Creek. These sub-networks have 16 to 29 stations on the area of few hundred km square. The soil moisture and soil temperature in the layer 0.00-0.05m are measured at each of the stations and the arithmetical average and area weighted average of the stations measurements are provided for each of the sub-networks. 
		Continent: Americas
		Country: USA
		Stations: 4
		Status: inactive
		Data Range: from 2002-06-01  to 2009-07-31
		Type: project
		Url: https://www.ars.usda.gov/
		Reference: Jackson, T.J., Cosh, M.H., Bindlish, R., Starks, P.J., Bosch, D.D., Seyfried, M.S., Goodrich, D.C., Moran, M.S., Validation of Advanced Microwave Scanning Radiometer Soil Moisture Products. IEEE Transactions on Geoscience and Remote Sensing. 48: 4256-4272, 2010, https://doi.org/10.1109/TGRS.2010.2051035;
		Variables: soil moisture, soil temperature, 
		Soil Moisture Depths: 0.00 - 0.05 m
		Soil Moisture Sensors: Hydraprobe Analog (2.5 Volt) area-weighted-average, Hydraprobe Analog (2.5 Volt) average, 

	VAS
		Abstract: Soil moisture network in Spain, University of Valencia, Climatology from Satellites Group and Jucar River Basin Authority
		Continent: Europe
		Country: Spain
		Stations: 3
		Status: running
		Data Range: from 2002-01-01 
		Type: meteo
		Url: http://nimbus.uv.es/

		Reference: We acknowledge the work of Yann H. Kerr and the VAS network team provided by the Climatology from Satellites Group (GCS) - University of Valencia in support of the ISMN;
		Variables: air temperature, soil moisture, soil temperature, 
		Soil Moisture Depths: 0.00 - 0.05 m
		Soil Moisture Sensors: Stevens Hydra Probe, ThetaProbe ML2X, 

	VDS
		Abstract: These networks have been installed to validate satellite soil moisture products over complex tropical regions.
		Continent: Asia
		Country: Myanmar
		Stations: 4
		Status: running
		Data Range: from 2017-06-01 
		Type: project
		Url: https://www.vandersat.com/
		Reference: We acknowledge the work of Ileen de Kat and Richard de Jeu as well as the VDS network team in support of the ISMN;

		Variables: soil moisture, soil temperature, 
		Soil Moisture Depths: 0.01 - 0.10 m, 0.10 - 0.10 m, 0.20 - 0.20 m
		Soil Moisture Sensors: GS1 Port 1, GS1 Port 2, GS1 Port 3, GS1 Port 4, GS1 Port 5, TEROS12, 

	WATERLINE
		Abstract: 
		Continent: Europe
		Country: Greece
		Stations: 3
		Status: running
		Data Range: from 2021-07-10 
		Type: Project
		Url: https://www.waterlineproject.eu/
		Reference: We acknowledge the work of Prof Alexandra Gemitzi, Stavros Stathopoulos and the entire WATERLINE team and their contributions in support of the ISMN.
		Variables: air temperature, precipitation, soil moisture, 
		Soil Moisture Depths: 0.05 - 0.05 m
		Soil Moisture Sensors: Sentek Drill and drop, 

	WEGENERNET
		Abstract: The WegenerNet Feldbach Region is a unique weather and climate observation network comprising more than 150 hydrometeorological stations measuring temperature, humidity, precipitation, and at 14 locations also wind speed and direction. Soil moisture and soil temperature are measured at 12 stations, which are part of the International Soil Moisture Network (ISMN). 


The stations are located in a tightly spaced grid within a core area of 22 km x 16 km centered near the city of Feldbach in southeastern Austria.
With about one station every two square-km (area of about 300 square-km in total), and each station with 5-min time sampling, the network provides fully automated regular measurements since January 2007.


************************************************************************************************************
IMPORTANT NOTE: All data is on version 8.0 For further details please see https://wegenernet.org/downloads/Fuchsberger-etal_2023_WPSv8-release-notes.pdf

************************************************************************************************************

		Continent: Europe
		Country: Austria
		Stations: 13
		Status: running
		Data Range: from 2007-01-01 
		Type: project
		Url: http://www.wegenernet.org/;http://www.wegcenter.at/wegenernet
		Reference: Fuchsberger, J., Kirchengast, G. & Kabas, T. (2021), WegenerNet high-resolution weather and climate data from 2007 to 2020, Earth Syst. Sci. Data, 13, 1307–1334, https://doi.org/10.5194/essd-13-1307-2021, 2021;

Kirchengast, G., Kabas, T., Leuprecht, A., Bichler, C. & Truhetz, H. (2014), ‘Wegenernet: A pioneering high-resolution network for monitoring weather and climate’, Bulletin of the American Meteorological Society 95, https://doi.org/10.1175/BAMS-D-11-00161.1;
		Variables: air temperature, precipitation, soil moisture, soil temperature, 
		Soil Moisture Depths: 0.20 - 0.20 m, 0.30 - 0.30 m
		Soil Moisture Sensors: Hydraprobe II, pF-Meter, 

	WIT-Network
		Abstract: 
		Continent: Asia
		Country: Pakistan
		Stations: 13
		Status: running
		Data Range: from 2022-04-04 
		Type: project
		Url: https://wit.lums.edu.pk/
		Reference: Dr. Abubakr Muhammad
		Variables: soil moisture, soil temperature, 
		Soil Moisture Depths: 0.10 - 0.10 m, 0.15 - 0.15 m
		Soil Moisture Sensors: EC5, WZ, 

	WSMN
		Abstract: The WSMN network is located in Wales, Great Britain, and consists of six stations. The datasets are collected by the Aberystwyth University and are available since September 2011.
		Continent: Europe
		Country: UK
		Stations: 8
		Status: running
		Data Range: from 2013-07-11 
		Type: project
		Url: http://www.aber.ac.uk/wsmn
		Reference: Petropoulos, G. P. & McCalmont, J. P. (2017), ‘An operational in situ soil moisture & soil temperature monitoring network for west wales, uk: The wsmn network’, Sensors 17(7), 1481, https://doi.org/10.3390/s17071481;

		Variables: soil moisture, soil temperature, 
		Soil Moisture Depths: 0.02 - 0.02 m, 0.05 - 0.05 m, 0.10 - 0.10 m
		Soil Moisture Sensors: CS615, CS616, CS655, 

	XMS-CAT
		Abstract: The soil monitoring network is a set of stations with continuous data recording of physics parameters (temperature and moisture of soil) and environmental parameters such as pluviometry, temperature, relative air humidity and solar radiation. Project started at Tremp basin’s, in the crops of vineyards, and it has been continued in high altitude vineyard form Pre-pyrenees and Pyrenees. The main features that set up the stations are the sensors installed into the soils and the basic operation elements, such as the acquisition data system, the power system and the data transmission system. Generally, there are 4 multi parametric sensors to 5, 20, 50 and 100 cm of depth, when parent material allows it, these sensors measure soil moisture and temperature. Each station also has environmental sensors such as a rain gauge, a pyranometer and a air temperature and relative humidity probes for the necessary comparison of the soil and environmental parameters.
		Continent: Europe
		Country: Spain
		Stations: 20
		Status: running
		Data Range: from 2016-08-01 
		Type: project
		Url: https://visors.icgc.cat/mesurasols/#9/42.1765/1.1132
		Reference: We acknowledge the work of Agnès Lladós and Lola Boquera as well as the XMS-CAT network team in support of the ISMN.
		Variables: air temperature, precipitation, soil moisture, soil temperature, 
		Soil Moisture Depths: 0.05 - 0.05 m, 0.10 - 0.10 m, 0.20 - 0.20 m, 0.30 - 0.30 m, 0.40 - 0.40 m, 0.50 - 0.50 m, 0.60 - 0.60 m, 0.70 - 0.70 m, 0.75 - 0.75 m, 1.00 - 1.00 m
		Soil Moisture Sensors: CS655, DeltaOHM, SoilVUE10, 


with chandeliers_front as (
select distinct safe_cast(itc.oms_id as string) omsid, itc.image_url, t.Taxonomy from  `analytics-online-data-sci-thd.mart.itc_weekly_results` itc
join `hd-datascience-np.common.CATGPENREL_taxonomy` t
on safe_cast(itc.oms_id as string) = t.omsid
where
--and itc.image_url like '%chandeliers%'
lower(t.Taxonomy ) not like '%featured%' and lower(t.Taxonomy ) like '%chandelier%' and lower(t.Taxonomy ) not like '%outdoor%' 
and itc.prediction_labels like '%front%'
),
width_dims as(
select omsid, ia.attributeid, ia.attributevalue width from `hd-personalization-prod.personalization.common_ItemAttributes` ia where ia.attributeid = 'fac03a00-67f9-402b-a121-fb9448fc552d' 
),
depth_dims as(
select omsid, ia.attributeid, ia.attributevalue depth from `hd-personalization-prod.personalization.common_ItemAttributes` ia where ia.attributeid = '5ad8d718-20da-4ce0-9872-fe7c84f24bb2' 
),
height_dims as(
select omsid, ia.attributeid, ia.attributevalue height from `hd-personalization-prod.personalization.common_ItemAttributes` ia where ia.attributeid = '542b2138-e766-4426-aa1a-250eadc74b50' 
),
canopy_dims as(
select omsid, ia.attributeid, ia.attributevalue canopy from `hd-personalization-prod.personalization.common_ItemAttributes` ia where ia.attributeid = 'ac4090da-33d3-419e-8ceb-15fa40b68b30' 
),
styles as(
select omsid, ia.attributeid ,a.displayname, ia.attributevalue as style from `hd-personalization-prod.personalization.common_ItemAttributes` ia join  `hd-personalization-prod.personalization.common_Attributes` a
on ia.attributeid = a.attribute_id 
where ia.attributeid = '2ffe2502-6e95-45a6-9123-3f4e5b9175d4'-- style
),
product_image as (
select omsid, ia.attributeid ,a.displayname, ia.attributevalue as image_guid from `hd-personalization-prod.personalization.common_ItemAttributes` ia join  `hd-personalization-prod.personalization.common_Attributes` a
on ia.attributeid = a.attribute_id 
where ia.attributeid = '645296e8-a910-43c3-803c-b51d3f1d4a89'-- product image  
),
top_ten_styles_info as (
select distinct chandeliers_front.omsid, image_guid, width, depth, height, canopy, style, Taxonomy taxonomy from chandeliers_front 
join width_dims on chandeliers_front.omsid = width_dims.omsid 
join depth_dims on chandeliers_front.omsid = depth_dims.omsid 
join height_dims on chandeliers_front.omsid = height_dims.omsid 
join canopy_dims on chandeliers_front.omsid = canopy_dims.omsid 
join styles on chandeliers_front.omsid = styles.omsid 
join product_image on chandeliers_front.omsid = product_image.omsid 
where styles.style in 
('Modern',
'Glam',
'Classic',
'Industrial',
'Classic,Transitional',
'Modern,Transitional',
'Rustic',
'Classic,Mediterranean',
'Classic,Mediterranean,Rustic,Southwestern',
'Classic,Modern,Transitional')
)

-- select count(*) cnts from top_ten_styles_info 
-- group by top_ten_styles_info.style 
-- order by cnts

-- select * from top_ten_styles_info 
-- limit 100


select distinct attributeid, displayname from(
select omsid, *  from `hd-personalization-prod.personalization.common_ItemAttributes` ia
join `hd-personalization-prod.personalization.common_Attributes` a on ia.attributeid = a.attribute_id         
where  lower(displayname) like '%canopy%' and lower(displayname) like '%in.%')
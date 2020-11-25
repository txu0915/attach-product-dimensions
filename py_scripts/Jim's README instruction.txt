Appliance Dimension Images


Github link: https://github.homedepot.com/jxa5pu1/appliance_dimensions/tree/master/dimension_images/prod

Github link to Model Mart utils: https://github.homedepot.com/jxa5pu1/model_mart_utils

GCS location of code run by MM: gs://hd-datascience-np-artifacts/jim/dimension_images

GCS location of output: gs://hd-datascience-np-data/dimension_images

Points of Contact: Amit Agarwal, Estelle Afshar, Harish Nair

Files in this directory:
    - code/: A copy of the code for this project.  This is just the main files used for production.  There are other scripts and results in the repo.

    - KnowledgeTransfer.pptx: a high level overview of the algorithm

    - appliance_sku_img.sql: a script to create the input data for the product image detector

Process to Run:
    - The process to run this project is a kind of involved, because the input keeps changing.  
    
    - At first, the assumption was that we needed to create a dimension image for each appliance given no extra input.
        - The script appliance_sku_img.sql contains within it a list of the appliance types that we want to generate dimension images for.  When it is run, it generates a file with 3 columns: appliance_type, oms_id, img_guid.  For each SKU, we will need to find which image to use as input for the dimension image algorithm.  This three column file should be put in the directory prod_img_test/00_appliance_sku_img.
    
        - Next, modify the file "prod_img_test/img_downloader.py" to point to this file, and run it.  This script does three things.  
            - It breaks the 00_appliance_sku_img apart into a separate file for each appliance type, and saves those in 02_inFiles/.
            
            - It generates a bash script named "run_product_img_detector.sh", which will run the product image detector twice per appliance type, once looking for a front-facing image and once looking for a side-facing image.
            
            - It downloads and saves each image listed in the 00_appliance_sku_img file.
    
        - Before actually running the product image detector, you need to supply the anchor images that are used to define what a good front/side image looks like for each appliance type.  Within the 03_anchors/ directory, you should create two folders per appliance type with names "APPLIANCETYPE_FRONT" and "APPLIANCETYPE_SIDE".  Save the anchor images within these directories.  You can (and probably should) have several examples per appliance type.
    
        - Run the "run_product_img_detector.sh" script.  This will save a front and side file per appliance type in the 05_outFiles/ directory.
    
        - The problem with the pipeline thus far is that the product image detector chooses what it thinks is the best match for each SKU, even if the SKU doesn't have an image of the designated type.  To address this, we train a random forest to take in the two predictions per SKU and tell us which to use.
    
        - Update the file "create_data_to_label.py" to point to the 00_appliance_sku_img file, and then run it. Within the 06_labelled folder, this will create three files per appliance type:
            - APPLIANCETYPE.csv: This is the full output of the product image detector for this appliance type.  It combines both the side and front facing image output.
           
            - APPLIANCETYPE.html: This is a visual representation of the output from the product image detector.  For each SKU, it shows the best side image and the best front image.
           
            - APPLIANCETYPE_labelled.csv: This is the file that you will have to fill in to train the model. 
    
        - Using the APPLIANCETYPE.html file, fill in the "img_to_use" column within APPLIANCETYPE_labelled.csv. If the side facing image for that SKU is good, enter "SIDE". If the side facing image isn't good but the front is, enter "FRONT". If neither is good, enter "neither".
    
        - Update the file "train_front_side_picker.py" to point to the 00_appliance_sku_img file, and then run it.  This file will train a random forest model to pick "SIDE", "FRONT", or "neither" for each SKU using the labelled data.  It will save the trained model in the 07_classifier/ directory.
    
        - Update the file "merge_front_side_imgs.py" to point to the 00_appliance_sku_img file, and then run it.  This will use the trained random forest model and apply it to the 06_labelled/APPLIANCETYPE.csv files.  The results are stored in 08_merged, one file per appliance type.  The output files have 4 columns: appliance_type, oms_id, img_guid, image_type (FRONT vs. SIDE).  SKUs where the model picks "neither" are left out.
    
    - All of the above was to figure out for each SKU, which image should we use.  Recently, it seems that the appliance team is willing to manually do that part for us.  If that continues to be the case, you will need to massage their Audit output into the correct format described above (appliance_type, oms_id, img_guid, img_type).  That is what the script parse_audit.py does, but might require changes if the audit output format changes.
    
    - Once you have the 4 column file described above, you can use that as the input to the Model Mart job.  The workflow is "dimension-images".  It assumes that the 4 column file will be in a directory named "merged_images".  The first component in the workflow pulls the length/width/height for each product and saves it in a directory "dimensions".  The second component will actually generate the images, saving them in output_images.  If you give it the flag, the second componenet will also save debug images. Each SKU will have an image saved with the oms_id as the filename.

Notes:
    - For now, to get the output onto the site I've simply been putting the results into a storage bucket which Amit has access to (gs://datascience-raw-data/appliance_dimension_images within analytics-online-data-sci project).  His team is taking the images and uploading them to IDM under a new "Line Art" attribute (the attribute id guid is de4ce5ef-0542-4324-aa34-49a76b5bccb5).

    - There are two config files:
        - dimension_config.json: This file lists the IDM attributes which should be used as the width/height/depth of the product.  You can provide a specific set of attributes for a specific product type, or there is a default.
      
        - image_drawing_config.json: This gives a bunch of parameters to control how the output images are drawn.
            - num_dim_decimal_pts: round the dimension's decimal value to at most this many decimal points
      
            - border_ratio: Adds a white border around the image to give more room to write the dimensions.  A value of 0.15 means that the border will be 15% the starting size of the image.
      
            - line_margin_ratio: Defines the margin between the computed edge of the product and where the dimension line is drawn
      
            - line_size_ratio: Controls how thick the dimension lines are drawn
      
            - font_file_loc: where the font .otf file is stored in GCS
      
            - font_size_ratio: Controls the font size
      
            - front: Controls aspects of drawing specifically the front-facing images
                - depth_angle_deg: The angle at which the depth line is drawn, in degrees.
      
                - start_depth_height_ratio: Start the depth line at this proportion of the product's height
      
                - end_depth_height_ratio: End the depth line at this proportion of the product's height
      
            - allow_swap_height_depth_appliance_types: Allow the height and depth values to be swapped for the specified product types

    - Work is currently being done on an OrangeSeeds validation front end.  The OrangeSeeds portal expects a json file as input which has a list of objects, one per sku.  A record for a sample SKU is shown below:

    {
        "oms_id": "205092837", 
        "dimension_image_gcs_location": "gs://hd-datascience-np-data/dimension_images/2018-12-31/output_images/Top_Freezer_Refrigerator/205092837.jpg", 
        "dimensions": {
            "width": "28", 
            "height": "64.75", 
            "depth": "31.875"
        }, 
        "selected_img_guid": "e9771433-c371-44f0-9795-61db99693c4a", 
        "selected_img_type": "FRONT", 
        "all_image_guids": [
            "e9771433-c371-44f0-9795-61db99693c4a", 
            "99da5dc6-18f2-426d-b2cb-00f6497bfd4d", 
            "861604cb-7418-49ca-af34-da734176679f", 
            "5404c417-4c97-4949-af80-beabd5d602e0", 
            "17d4da19-9a41-4896-a7e9-4c7e5df9ec01"
        ]
    }

    The file prod_img_test/orange_seeds_input.py is a rough script to generate a file like this, but it still needs work.  In addition, Orange Seeds currently doesn't have the access necessary to read directly from the Model Mart storage bucket. Until it does, we are uploading the images from model mart storage to our Datascience storage and then making the images publicly available from there.  The steps are
        1. Copy the images from Model Mart storage to Datascience storage.  Use a command like

            gsutil -m cp -r gs://hd-datascience-np-data/dimension_images/<DATE>/output_images/* gs://ds-jim/dimension_images/

        2. Make the images public with a command like 

            gsutil -m acl ch -r -u AllUsers:R gs://ds-jim/dimension_images/

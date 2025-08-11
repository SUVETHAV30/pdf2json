import os
import json
import sys
import functools
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import fitz # Added for image extraction
from PIL import Image # Added for image processing
import io # Added for image processing
import functools # Added for image processing
from mistralai import Mistral
from groq import Groq

from models import (
    ProcessingType, ProductInfo, ExtractionResult, 
    ProductSpecification, ProductFeature, ImageMetadata
)
from text_engine import TextExtractionEngine
from algorithms import TableExtractionAlgorithm

from logger import setup_logger
logger= setup_logger(name="pdf")


class PDFProcessor:
    """Main PDF processing orchestrator"""
    
    def __init__(self, output_base: str = "output"):
        self.output_base = output_base
        self.text_engine = TextExtractionEngine()

    def get_images_from_pdf(self, pdf_path: str) -> Dict[str, Any]:

        name = pdf_path.split("/")[-1].split(".")[0]
        doc = fitz.open(pdf_path)
        image_index = 0
        feature_images = {}
        product_image = {}

        for page_index, page in enumerate(doc):
            w, h = page.rect.width, page.rect.height
            logger.info(f"Page {page_index + 1}: Width = {w} pt, Height = {h} pt")

            for img_index, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                bbox = page.get_image_bbox(img)
                base_image = doc.extract_image(xref)

                width = base_image["width"]
                height = base_image["height"]
                image_bytes = base_image["image"]

                if height < 100 or width < 100:
                    print(f"Skipping image {img_index+1} on page {page_index+1} due to small size: {width}x{height}")
                    continue

                if int(bbox.x0) > w/2 and int(bbox.y0) > (h/2):

                    feature_images[image_index] = {
                        "name": name,
                        "bbox": bbox,
                        "image_bytes": image_bytes,
                        "width": width,
                        "height": height,
                    }  
                elif int(bbox.x0) < 10 or int(bbox.y0) < 10 :
                    
                    product_image[10] = {
                        "name": name,
                        "bbox": bbox,
                        "image_bytes": image_bytes,
                        "width": width,
                        "height": height,
                    } 
                image_index += 1
        
        def compare_items(a, b):
            bbox_a = a[1]["bbox"]
            bbox_b = b[1]["bbox"]

            diff_x0 = abs(bbox_a.x0 - bbox_b.x0)

            if diff_x0 > 5:
                # If x0 difference is large, sort by x0
                return -1 if bbox_a.x0 < bbox_b.x0 else 1
            else:
                # If x0 is close, sort by y0
                return -1 if bbox_a.y0 < bbox_b.y0 else 1

        # Apply custom comparator
        sorted_feature_images = sorted(
            feature_images.items(),
            key=functools.cmp_to_key(compare_items)
        )


        renumbered_feature_images = {
            new_idx: data for new_idx, (_, data) in enumerate(sorted_feature_images)
        }
        renumbered_feature_images.update(product_image)
        extracted_img_paths = self.save_images(renumbered_feature_images)
        
        logger.info("---------------------")
        logger.info(extracted_img_paths)
        logger.info(len(extracted_img_paths))
        logger.info("---------------------")

        return extracted_img_paths , len(extracted_img_paths)
        
        
    def save_images(self,feature_images, output_base="extracted_images"):
        
        # Create output folder if it doesn't exist
        output_dir = os.path.join(output_base)
        os.makedirs(output_dir, exist_ok=True)
                    
        img_extracted_paths= []
        for image_index, data in feature_images.items():
            image_bytes = data["image_bytes"]

            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_bytes))

            # Save as JPEG
            output_path = os.path.join(output_dir, f"{image_index}.jpg")
            image.save(output_path, format="JPEG")
            img_extracted_paths.append(output_path)
            
        return img_extracted_paths


    def post_process(self,raw_text,structured_text) :
                
        # Extract tables
        tables = [] # self.extract_tables_structured(ocr_response)

        # Build specifications from structured content and tables
        specifications = [
            ProductSpecification(**s)
            for s in structured_text.get("specifications", [])
        ]
        
        # Add specifications from tables
        for table in tables:
            if table.table_type == "specifications":
                for row in table.rows:
                    if len(row) >= 2:
                        specifications.append(ProductSpecification(
                            label=row[0], 
                            value=row[1],
                            unit=row[2] if len(row) > 2 else None
                        ))

        # Build features
        features = [ProductFeature(**f) for f in structured_text.get("features", [])]

        # Create product info
        product_info = ProductInfo(
            product_name=structured_text.get("product_name", ""),
            product_description=structured_text.get("product_description", ""),
            model_number=", ".join(structured_text.get("model_number")) if isinstance(structured_text.get("model_number"), list) else structured_text.get("model_number"),
            brand=structured_text.get("brand"),
            specifications=specifications,
            features=features,
            tables=tables,
        )

        logger.info("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        logger.info(product_info)
        logger.info("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        

        # Convert to dictionar
        product_info_dict = product_info.dict() if hasattr(product_info, 'dict') else product_info.__dict__

        # Define save path
        output_path = "output/product_info.json"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save to JSON
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(product_info_dict, f, ensure_ascii=False, indent=4)

        logger.info(f"Saved product_info to {output_path}")

        
        return product_info_dict

    def detect_category_from_text(self, raw_text: str) -> str:
        """Intelligently detect category from text content"""
        text_lower = raw_text.lower()
        
        # Common category patterns
        category_patterns = {
            "brush cutter": ["brush cutter", "brushcutter", "brush-cutter"],
            "chainsaw": ["chainsaw", "chain saw", "chain-saw"],
            "lawn mower": ["lawn mower", "lawnmower", "mower"],
            "trimmer": ["trimmer", "string trimmer", "line trimmer"],
            "blower": ["blower", "leaf blower", "garden blower"],
            "hedge trimmer": ["hedge trimmer", "hedge cutter"],
            "power tool": ["power tool", "power equipment"],
            "garden tool": ["garden tool", "garden equipment"],
            "outdoor equipment": ["outdoor equipment", "outdoor power equipment"],
            "construction tool": ["construction tool", "construction equipment"],
            "agricultural equipment": ["agricultural", "farm equipment", "farming"],
            "forestry equipment": ["forestry", "logging equipment"],
            "landscaping equipment": ["landscaping", "landscape equipment"]
        }
        
        # Check for category patterns
        for category, patterns in category_patterns.items():
            for pattern in patterns:
                if pattern in text_lower:
                    return category.title()
        
        # Check for brand-specific categories
        if "maruyama" in text_lower:
            if "brush" in text_lower or "cutter" in text_lower:
                return "Brush Cutter"
            elif "chainsaw" in text_lower:
                return "Chainsaw"
            elif "blower" in text_lower:
                return "Blower"
        
        # Check for engine types that might indicate category
        if "cc" in text_lower or "engine" in text_lower:
            if "brush" in text_lower or "cutter" in text_lower:
                return "Brush Cutter"
            elif "chain" in text_lower:
                return "Chainsaw"
        
        return "Power Equipment"  # Default category

    def convert_to_acf_format(self, product_info_dict: Dict[str, Any], saved_image_paths: List[str], raw_text: str) -> List[Dict[str, Any]]:
        """
        Convert extracted product info to ACF format
        """
        # Get main image (first image)
        main_image = {"url": saved_image_paths[0]} if saved_image_paths else {"url": ""}
        
        # Get thumbnails (remaining images, up to 4)
        thumbnails = []
        for path in saved_image_paths[1:5]:  # Skip first (main image), take up to 4
            thumbnails.append({"url": path})
        
        # If we don't have enough thumbnails, pad with main image or empty
        while len(thumbnails) < 4:
            if saved_image_paths:
                thumbnails.append({"url": saved_image_paths[0]})  # Use main image as fallback
            else:
                thumbnails.append({"url": ""})

        # Create detailed HTML description
        html_parts = []
        
        # Add product description
        if product_info_dict.get('product_description'):
            html_parts.append(f"<p>{product_info_dict['product_description']}</p>")
        
        # Add brand and model info
        brand = product_info_dict.get('brand', '')
        model = product_info_dict.get('model_number', '')
        if brand or model:
            html_parts.append(f"<p><strong>Brand:</strong> {brand} | <strong>Model:</strong> {model}</p>")
        
        # Add key specifications as HTML
        specs = product_info_dict.get('specifications', [])
        if specs:
            html_parts.append("<h3>Key Specifications:</h3><ul>")
            for spec in specs[:5]:  # Limit to top 5 specs for detailed description
                spec_dict = spec if isinstance(spec, dict) else spec.__dict__
                if spec_dict.get('value'):  # Only include specs with values
                    unit = f" {spec_dict['unit']}" if spec_dict.get('unit') else ""
                    html_parts.append(f"<li><strong>{spec_dict['label']}:</strong> {spec_dict['value']}{unit}</li>")
            html_parts.append("</ul>")
        
        # Add features as HTML
        features = product_info_dict.get('features', [])
        if features:
            html_parts.append("<h3>Key Features:</h3><ul>")
            for feature in features:
                feature_dict = feature if isinstance(feature, dict) else feature.__dict__
                html_parts.append(f"<li><strong>{feature_dict['name']}:</strong> {feature_dict['description']}</li>")
            html_parts.append("</ul>")

        detailed_description = "".join(html_parts) if html_parts else "<p>Product details coming soon...</p>"

        # Convert features to ACF format
        acf_features = []
        for i, feature in enumerate(features):
            feature_dict = feature if isinstance(feature, dict) else feature.__dict__
            # Assign images cyclically to features
            image_url = saved_image_paths[i % len(saved_image_paths)] if saved_image_paths else ""
            
            acf_feature = {
                "title": feature_dict.get('name', ''),
                "description": feature_dict.get('description', ''),
                "image": {"url": image_url}
            }
            acf_features.append(acf_feature)

        # Convert specifications to ACF format
        acf_specs = []
        for spec in specs:
            spec_dict = spec if isinstance(spec, dict) else spec.__dict__
            if spec_dict.get('value'):  # Only include specs with values
                # Combine value and unit if unit exists
                value = spec_dict['value']
                if spec_dict.get('unit'):
                    value = f"{value} {spec_dict['unit']}"
                
                acf_spec = {
                    "label": spec_dict.get('label', ''),
                    "value": value
                }
                acf_specs.append(acf_spec)

        # Create the ACF formatted product
        acf_product = {
            "title": f"{product_info_dict.get('product_name', 'Unknown')}",
            "acf": {
                "product_name": product_info_dict.get('product_name', ''),
                "category": self.detect_category_from_text(product_info_dict.get("product_description", "") + " " + product_info_dict.get("product_name", "")),  # Use the new detection logic
                "main_image": main_image,
                "thumbnails": thumbnails,
                "product_description": product_info_dict.get('product_description', ''),
                "detailed_description": detailed_description,
                "rating": 4.5,  # Default rating
                "review_count": 10,  # Default review count
                "features": acf_features,
                "specifications": acf_specs
            }
        }
        
        # Return as list format as per your requirements
        return [acf_product]
        

    def process_pdf(self, pdf_path: str, processing_type: ProcessingType = ProcessingType.COMBINED) -> Dict[str, Any]:
        """Main processing method with configurable extraction type"""
        
        pdf_path_obj = Path(pdf_path)
        if not pdf_path_obj.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        saved_image_paths = []
        no_of_imgs = 0
        
        # Image processing
        if processing_type == "combined":
            logger.info("Extracting images...")
            saved_image_paths,no_of_imgs = self.get_images_from_pdf(pdf_path)
            
            logger.info("==============")
            logger.info("Saved paths ... ")
            logger.info(saved_image_paths)
            
        raw_text = ""
        structured_text = {}
        # Initialize for clarity
        
        # Text processing
        if processing_type == "combined":
            logger.info("Extracting text...")
            raw_text = self.text_engine.extract_text_from_pdf(pdf_path_obj)
            
            logger.info("Structuring text...")
            structured_text = self.text_engine.structure_text_to_json(raw_text,no_of_imgs)
            
            product_info_dict = self.post_process(raw_text,structured_text)
            logger.info("--------------------------------------")
            logger.info(f"Product info {product_info_dict}")
            logger.info("---------------------------------------")

            # NEW: Convert to ACF format
            logger.info("Converting to ACF format...")
            acf_formatted_data = self.convert_to_acf_format(product_info_dict, saved_image_paths, raw_text)
            
            # Save ACF formatted data
            acf_output_path = "output/acf_product_data.json"
            os.makedirs("output", exist_ok=True)
            with open(acf_output_path, 'w') as f:
                json.dump(acf_formatted_data, f, indent=2)
            
            logger.info(f"ACF formatted data saved to {acf_output_path}")
            logger.info("ACF formatted result:")
            logger.info(json.dumps(acf_formatted_data, indent=2))

            return {
                "original_data": product_info_dict,
                "acf_formatted_data": acf_formatted_data,
                "saved_image_paths": saved_image_paths,
                "status": "success"
            }

        return {"status": "no processing type specified"}
    
    
    def _build_product_info(self, structured_text: Dict, image_results: Dict, raw_text: str) -> ProductInfo:
        """Build ProductInfo object from extracted data"""
        tables = TableExtractionAlgorithm.extract_tables_from_text(raw_text) if raw_text else []
        
        specifications = [
            ProductSpecification(
                label=spec.get("label", ""), 
                value=spec.get("value", ""), 
                unit=spec.get("unit", None)
            )
            for spec in structured_text.get("specifications", [])
        ]
        
        # Use a list to maintain order and a set to track seen items for deduplication
        final_specifications = []
        seen_specs = set()

        # Add specifications from structured_text first, maintaining their order
        for spec in specifications:
            spec_tuple = (spec.label, spec.value, spec.unit)
            if spec_tuple not in seen_specs:
                final_specifications.append(spec)
                seen_specs.add(spec_tuple)

        # Add specifications from tables, maintaining their order relative to table parsing
        for table in tables:
            if table.table_type == "specifications":
                for row in table.rows:
                    if len(row) >= 2:
                        label = row[0]
                        value = row[1]
                        unit = row[2] if len(row) > 2 else None
                        spec_tuple = (label, value, unit)
                        if spec_tuple not in seen_specs:
                            final_specifications.append(ProductSpecification(label=label, value=value, unit=unit))
                            seen_specs.add(spec_tuple)
        
        features = [
            ProductFeature(**feat) 
            for feat in structured_text.get("features", [])
        ]
        
        images = [
            ImageMetadata(**img) if isinstance(img, dict) else img
            for img in image_results.get("images", [])
        ]
        
        return ProductInfo(
            product_name=structured_text.get("product_name", ""),
            product_description=structured_text.get("product_description", ""),
            model_number=structured_text.get("model_number", ""),
            brand=structured_text.get("brand", ""),
            category=structured_text.get("category", ""),
            specifications=final_specifications, # Use the order-preserved, deduplicated list
            features=features,
            tables=tables,
            images=images,
            raw_text=raw_text
        )
    
    def save_results(self, result: ExtractionResult, output_dir: str = None) -> Dict[str, str]:
        """Save extraction results in multiple formats"""
        if output_dir is None:
            output_dir = self.output_base
        
        os.makedirs(output_dir, exist_ok=True)
        
        source_name = Path(result.metadata["source_pdf"]).stem
        base_filename = f"{source_name}_{result.processing_type}"
        
        saved_files = {}
        
        # Save structured JSON
        structured_path = os.path.join(output_dir, f"{base_filename}_structured.json")
        with open(structured_path, 'w', encoding='utf-8') as f:
            json.dump(result.model_dump(), f, indent=2, ensure_ascii=False, default=str)
        saved_files["structured"] = structured_path
        
        # Save frontend format - this will now include base64 encoded images
        frontend_data = self._convert_to_frontend_format(result)
        frontend_path = os.path.join(output_dir, f"{base_filename}_frontend.json")
        with open(frontend_path, 'w', encoding='utf-8') as f:
            json.dump(frontend_data, f, indent=2, ensure_ascii=False, default=str)
        saved_files["frontend"] = frontend_path
        
        # Save raw text if available
        if result.products and result.products[0].raw_text:
            text_path = os.path.join(output_dir, f"{base_filename}_raw_text.txt")
            with open(text_path, 'w', encoding='utf-8') as f:
                f.write(result.products[0].raw_text)
            saved_files["raw_text"] = text_path
        
        return saved_files
    
    def _convert_to_frontend_format(self, result: ExtractionResult) -> Dict[str, Any]:
        """Convert extraction result to frontend-friendly format"""
        if not result.products:
            return {"products": []}
        
        product = result.products[0]
        images = product.images
        
        # In this updated version, images are embedded as base64 in the JSON.
        # The 'local_path' and 'thumbnails' will be empty or handle the base64 data directly.
        
        frontend_product = {
            "product_name": product.product_name,
            "product_description": product.product_description,
            "category": product.category,
            "brand": product.brand,
            "model_number": product.model_number,
            "rating": "4.5",
            "reviewCount": "128",
            "detailedDescription": "",
            "features": [
                {"name": f.name, "description": f.description or ""}
                for f in product.features
            ],
            "specifications": [
                {"label": s.label, "value": f"{s.value} {s.unit or ''}".strip()}
                for s in product.specifications
            ],
            "mainImage": "", # No main image path as images are embedded
            "thumbnails": [], # No thumbnails path as images are embedded
            "images_folder": "", # No image folder as images are embedded
            "image_details": [img.model_dump() for img in images]
        }
        
        return {
            "category": [product.category] if product.category else [],
            "products": [frontend_product]
        }

    def print_summary(self, result: ExtractionResult):
        """Print a comprehensive summary of extraction results"""
        print("\n" + "=" * 60)
        print(f"PDF PROCESSING SUMMARY - {result.processing_type.upper()}")
        print("=" * 60)
        
        print(f"Source: {result.metadata['source_pdf']}")
        print(f"Processed: {result.metadata['extraction_timestamp']}")
        print(f"Type: {result.processing_type}")
        
        if result.products:
            product = result.products[0]
            print(f"\nProduct: {product.product_name}")
            print(f"Category: {product.category}")
            print(f"Brand: {product.brand}")
            print(f"Features: {len(product.features)}")
            print(f"Specifications: {len(product.specifications)}")
            print(f"Images: {len(product.images)}")
            print(f"Tables: {len(product.tables)}")
            
            if product.raw_text:
                print(f"Text Length: {len(product.raw_text)} characters")
        
        print("=" * 60)
        
if __name__ == "__main__":
    process = PDFProcessor()
    result = process.process_pdf(pdf_path="data/test.pdf", processing_type="combined")
    print("Processing completed:", result["status"])
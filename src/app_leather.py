import os
import customtkinter as ctk
from tkinter import filedialog
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms as transforms

# 1. Initialize Advanced Theme
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

class LeatherClassifierApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Advanced Leather Texture Classifier")
        self.geometry("1000x650")
        self.minsize(900, 600)
        
        # Grid Configuration (Split into Left and Right panels)
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        # System Variables
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.class_names = []
        self.dataset_dir = "dataset_train"
        self.result_image_references = [] # Keep references to prevent garbage collection
        
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

        self.setup_ui()
        self.load_system_data()

    def setup_ui(self):
        # ================= LEFT PANEL (Input) =================
        self.left_frame = ctk.CTkFrame(self, corner_radius=15)
        self.left_frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        self.left_frame.grid_columnconfigure(0, weight=1)

        self.input_title = ctk.CTkLabel(self.left_frame, text="Input Analysis", font=ctk.CTkFont(size=20, weight="bold"))
        self.input_title.grid(row=0, column=0, pady=(20, 10))

        self.image_display_frame = ctk.CTkFrame(self.left_frame, width=350, height=350, fg_color="gray15")
        self.image_display_frame.grid(row=1, column=0, pady=10)
        self.image_display_frame.pack_propagate(False)

        self.main_image_label = ctk.CTkLabel(self.image_display_frame, text="No Image Selected", text_color="gray50")
        self.main_image_label.pack(expand=True)

        self.upload_btn = ctk.CTkButton(self.left_frame, text="Load Image", command=self.upload_and_predict, height=45, font=ctk.CTkFont(size=15, weight="bold"))
        self.upload_btn.grid(row=2, column=0, pady=20)

        self.status_label = ctk.CTkLabel(self.left_frame, text="Initializing system...", text_color="gray70")
        self.status_label.grid(row=3, column=0, pady=(10, 20))

        # ================= RIGHT PANEL (Results) =================
        self.right_frame = ctk.CTkFrame(self, corner_radius=15, fg_color="transparent")
        self.right_frame.grid(row=0, column=1, padx=(0, 20), pady=20, sticky="nsew")
        self.right_frame.grid_columnconfigure(0, weight=1)

        self.result_title = ctk.CTkLabel(self.right_frame, text="Top 3 Matches", font=ctk.CTkFont(size=20, weight="bold"))
        self.result_title.grid(row=0, column=0, pady=(0, 15), sticky="w")

        # Create 3 rows for Top 3 results
        self.result_rows = []
        for i in range(3):
            row_frame = ctk.CTkFrame(self.right_frame, corner_radius=10, fg_color="gray15")
            row_frame.grid(row=i+1, column=0, pady=(0, 15), sticky="ew")
            row_frame.grid_columnconfigure(1, weight=1)

            # Thumbnail Label
            thumb_label = ctk.CTkLabel(row_frame, text="Image", width=100, height=100, fg_color="gray20", corner_radius=5)
            thumb_label.grid(row=0, column=0, padx=10, pady=10)

            # Text Info Frame
            info_frame = ctk.CTkFrame(row_frame, fg_color="transparent")
            info_frame.grid(row=0, column=1, padx=10, pady=10, sticky="w")

            rank_label = ctk.CTkLabel(info_frame, text=f"Rank {i+1}", font=ctk.CTkFont(size=14, weight="bold"), text_color="#1f6aa5")
            rank_label.pack(anchor="w")

            class_label = ctk.CTkLabel(info_frame, text="---", font=ctk.CTkFont(size=18, weight="bold"))
            class_label.pack(anchor="w", pady=(5, 0))

            score_label = ctk.CTkLabel(info_frame, text="Confidence: ---%", font=ctk.CTkFont(size=14), text_color="gray60")
            score_label.pack(anchor="w")

            self.result_rows.append({
                "thumb": thumb_label,
                "class": class_label,
                "score": score_label
            })

    def load_system_data(self):
        model_path = "best_leather_model_val.pth"
        
        if not os.path.exists(self.dataset_dir):
            self.status_label.configure(text="Error: 'dataset_train' folder missing", text_color="#ff4a4a")
            return
            
        self.class_names = sorted(os.listdir(self.dataset_dir))

        if not os.path.exists(model_path):
            self.status_label.configure(text="Error: Model weights missing", text_color="#ff4a4a")
            return

        try:
            self.model = models.resnet50(weights=None)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, len(self.class_names))
            self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
            self.model.to(self.device)
            self.model.eval()
            self.status_label.configure(text="System Ready", text_color="#2fa572")
        except Exception as e:
            self.status_label.configure(text=f"Model Load Failed: {str(e)}", text_color="#ff4a4a")

    def get_reference_image(self, class_name):
        class_dir = os.path.join(self.dataset_dir, class_name)
        if os.path.exists(class_dir):
            files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            if files:
                return os.path.join(class_dir, files[0])
        return None

    def upload_and_predict(self):
        if self.model is None:
            self.status_label.configure(text="Cannot predict: Model not loaded", text_color="#ff4a4a")
            return

        file_path = filedialog.askopenfilename(
            title="Select Leather Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )

        if not file_path:
            return

        # Update Main Image Display
        display_img = Image.open(file_path)
        ctk_image = ctk.CTkImage(light_image=display_img, dark_image=display_img, size=(330, 330))
        self.main_image_label.configure(image=ctk_image, text="")
        
        self.status_label.configure(text="Analyzing features...", text_color="white")
        self.update()

        try:
            # Inference
            input_img = Image.open(file_path).convert('RGB')
            img_tensor = self.transform(input_img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.model(img_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                
                # Get Top 3 results
                top_probs, top_indices = torch.topk(probabilities, 3)

            self.result_image_references.clear()

            # Update Top 3 UI
            for i in range(3):
                idx = top_indices[i].item()
                prob = top_probs[i].item() * 100
                predicted_class = self.class_names[idx]
                
                row_ui = self.result_rows[i]
                row_ui["class"].configure(text=predicted_class)
                row_ui["score"].configure(text=f"Confidence: {prob:.2f}%")

                # Load and display reference image
                ref_img_path = self.get_reference_image(predicted_class)
                if ref_img_path:
                    ref_img = Image.open(ref_img_path)
                    ref_ctk_img = ctk.CTkImage(light_image=ref_img, dark_image=ref_img, size=(100, 100))
                    row_ui["thumb"].configure(image=ref_ctk_img, text="")
                    self.result_image_references.append(ref_ctk_img) # Save reference
                else:
                    row_ui["thumb"].configure(image="", text="No Image")

            self.status_label.configure(text="Analysis Complete", text_color="#2fa572")
            
        except Exception as e:
            self.status_label.configure(text="Error during analysis", text_color="#ff4a4a")
            print(f"Error details: {e}")

if __name__ == "__main__":
    app = LeatherClassifierApp()
    app.mainloop()
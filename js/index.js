import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "LoadImageWithFilename.Browse",
    async nodeCreated(node) {
        if (node.comfyClass !== "LoadImageWithFilename") return;
        
        // Create file input
        const fileInput = document.createElement("input");
        fileInput.type = "file";
        fileInput.accept = "image/*";
        fileInput.style.display = "none";
        
        // Find the image input widget
        const imageWidget = node.widgets.find(w => w.name === "image");
        if (!imageWidget) return;
        
        // Create browse button using addDOMWidget
        const button = document.createElement("button");
        button.textContent = "📂 Browse";
        button.style.cssText = "margin-top: 8px; padding: 6px 12px; cursor: pointer; background: #333; color: #fff; border: 1px solid #555; border-radius: 4px; width: 100%;";
        
        fileInput.addEventListener("change", async (e) => {
            const file = e.target.files[0];
            if (!file) return;
            
            // Try to get full path - works in Electron
            let filePath = file.path;
            
            // If no path from Electron
            if (!filePath || filePath.startsWith("C:\\fakepath") || !filePath.includes("\\")) {
                console.warn("Cannot get full file path in browser mode");
                filePath = file.name;
            }
            
            imageWidget.value = filePath;
            
            // Trigger widget callback
            node.setDirty(true);
            if (imageWidget.callback) {
                imageWidget.callback();
            }
            
            // Force resize
            node.setSize(node.computeSize());
        });
        
        button.addEventListener("click", (e) => {
            e.preventDefault();
            fileInput.click();
        });
        
        // Add button to node
        node.addDOMWidget("browse_btn", "button", button);
        
        // Add hidden file input to DOM
        document.body.appendChild(fileInput);
    }
});

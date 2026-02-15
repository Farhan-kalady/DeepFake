document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const previewContainer = document.getElementById('previewContainer');
    const previewImage = document.getElementById('previewImage');
    const previewVideo = document.getElementById('previewVideo');
    const uploadForm = document.getElementById('uploadForm');
    const scanBtn = document.getElementById('scanBtn');
    const btnText = document.getElementById('btnText');
    const btnSpinner = document.getElementById('btnSpinner');

    // Drag and Drop Logic
    if (dropZone && fileInput) {
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, preventDefaults, false);
        });

        function preventDefaults(e) {
            e.preventDefault();
            e.stopPropagation();
        }

        ['dragenter', 'dragover'].forEach(eventName => {
            dropZone.addEventListener(eventName, () => {
                dropZone.classList.add('drag-over');
            }, false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, () => {
                dropZone.classList.remove('drag-over');
            }, false);
        });

        dropZone.addEventListener('drop', (e) => {
            const dt = e.dataTransfer;
            const files = dt.files;
            fileInput.files = files;
            handleFiles(files);
        }, false);

        fileInput.addEventListener('change', function () {
            handleFiles(this.files);
        });
    }

    function handleFiles(files) {
        if (files.length > 0 && previewContainer && previewImage && previewVideo) {
            const file = files[0];
            const reader = new FileReader();

            if (file.type.startsWith('image/')) {
                reader.onload = (e) => {
                    previewImage.src = e.target.result;
                    previewImage.classList.remove('hidden');
                    previewVideo.classList.add('hidden');
                    previewContainer.classList.remove('hidden');
                };
                reader.readAsDataURL(file);
            } else if (file.type.startsWith('video/')) {
                const url = URL.createObjectURL(file);
                previewVideo.src = url;
                previewVideo.classList.remove('hidden');
                previewImage.classList.add('hidden');
                previewContainer.classList.remove('hidden');
            }
        }
    }


    // Form Submission UI
    if (uploadForm) {
        uploadForm.addEventListener('submit', () => {
            scanBtn.disabled = true;
            btnText.innerText = 'Analyzing Content...';
            btnSpinner.classList.remove('hidden');
            scanBtn.classList.add('opacity-75', 'cursor-not-allowed');
        });
    }

    // Heatmap Toggle Logic
    const heatmapToggle = document.getElementById('heatmapToggle');
    const heatmapOverlay = document.querySelector('.heatmap-overlay');
    const originalImage = document.querySelector('.original-image');

    if (heatmapToggle && heatmapOverlay) {
        heatmapToggle.addEventListener('change', function () {
            if (this.checked) {
                heatmapOverlay.classList.remove('opacity-0');
                heatmapOverlay.classList.add('opacity-100');
                if (originalImage) originalImage.style.filter = 'brightness(0.7)';
            } else {
                heatmapOverlay.classList.add('opacity-0');
                heatmapOverlay.classList.remove('opacity-100');
                if (originalImage) originalImage.style.filter = 'none';
            }
        });
    }

});

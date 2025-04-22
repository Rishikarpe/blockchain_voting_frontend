const express = require('express');
const multer = require('multer');
const path = require('path');
const bodyParser = require('body-parser');
const app = express();

// Setup storage engine for Multer (to save files)
const storage = multer.diskStorage({
    destination: function (req, file, cb) {
        cb(null, 'uploads/'); // Folder to save uploaded files
    },
    filename: function (req, file, cb) {
        cb(null, Date.now() + path.extname(file.originalname)); // Adding timestamp to avoid name collision
    }
});
const upload = multer({ storage: storage });

// Middleware to parse JSON bodies
app.use(bodyParser.json());
app.use(express.static('public')); // Serve static files (like the HTML form)

// POST route for handling e-KYC form submission
app.post('/verify-kyc', upload.single('document'), (req, res) => {
    const { name } = req.body;
    const document = req.file;  // The file uploaded

    // For now, just simulate a basic check:
    if (!document) {
        return res.status(400).json({ message: 'No document uploaded.' });
    }

    // Simulate document validation
    // Ideally, you would integrate a real document verification system here.
    // For now, just return the file name and user details
    res.json({
        message: 'e-KYC document uploaded successfully.',
        name: name,
        document: document.filename
    });
});

// Serve static files (for testing)
app.use(express.static('public'));

// Start server on port 3000
app.listen(3000, () => {
    console.log('Server running on http://localhost:3000');
});

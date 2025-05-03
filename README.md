# LISA: Lab Inventory Support Agent

**LISA** is a robotic assistant designed to maintain synchronization between physical lab inventory and digital lab records (LIMS), helping researchers keep their workspaces efficient, accurate, and clean. Developed at Carnegie Mellon University in partnership with LLNL.

## 🧠 What is LISA?

LISA is a symbiotic lab robot built to:
- Track and retrieve items using computer vision.
- Compare Safety Data Sheets and protocols with LIMS.
- Aid in lab setup and cleanup through robust object recognition and manipulation.
- Continuously update its internal inventory model even when items are moved by humans.

> “LISA knows where everything is—and helps you put it back.”  
> — *Capstone Presentation, 2025*
## 🔧 Key Capabilities

- **Easy Calibration**: Interactive visual setup with lab technicians.
- **Vision System**: Uses AprilTags now, aiming for YOLO/SAM in future.
- **Path Planning**: Avoids robot arm self-collision via custom interpolation and forbidden zone detection.
- **Protocol Setup**: Reads experimental PDFs and cross-references inventory needs with LIMS.
- **Analytics**: 97.5% retrieval success rate for small/medium items; 15.2s average retrieval time.

## 📊 Performance Metrics

| Metric | Value |
|--------|-------|
| Retrieval Accuracy | 97.5% |
| Protocol Item Matching | 100% |
| Retrieval Time | 15.2 seconds |
| Avg. Gripper Deviation | ±3.02 cm |

## 🧪 Current Limitations

- Vision struggles with transparent objects.
- Gripper can't handle large tipboxes/plates.
- AprilTags are not scalable for production—need segmentation-based vision.

## 🤝 Collaborators

- Jon Potter ([jonpot.com](https://www.jonpot.com))  
- Zhishan "Mercury" Liu  
- In collaboration with **LLNL Mercury Team**

## 📬 Contact

📧 jpotter2@cs.cmu.edu  
🌐 [www.jonpot.com](https://www.jonpot.com)

---

*Capstone Presentation, Carnegie Mellon University, 2025. For research/demo purposes only.*

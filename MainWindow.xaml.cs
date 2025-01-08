using System;
using System.Windows;
using OpenCvSharp;
using System.Linq;
using System.IO;
using Microsoft.Win32;
using System.Drawing;
using OpenCvSharp.Dnn;
using System.Threading;
using System.Threading.Tasks;

namespace VidAi_WPF
{
    public partial class MainWindow : System.Windows.Window
    {
        private string videoPath;
        private VideoCapture capture;
        private CancellationTokenSource cancellationTokenSource;
        private Task videoProcessingTask;
        private bool isVideoPlaying = false;  

        public MainWindow()
        {
            InitializeComponent();
        }

        private void SelectFile_Click(object sender, RoutedEventArgs e)
        {
            OpenFileDialog openFileDialog = new OpenFileDialog();
            openFileDialog.Filter = "Video Files (*.mp4;*.avi)|*.mp4;*.avi";
            if (openFileDialog.ShowDialog() == true)
            {
                videoPath = openFileDialog.FileName;
                StartObjectDetection(videoPath);
            }
        }

        private void StartObjectDetection(string videoPath)
        {
            StopCurrentVideoProcessing();

            cancellationTokenSource = new CancellationTokenSource();
            var cancellationToken = cancellationTokenSource.Token;

            videoProcessingTask = Task.Run(() =>
            {
                try
                {
                    string modelWeights = "yolov3-tiny.weights";
                    string modelConfig = "yolov3-tiny.cfg";
                    string classNamesFile = "coco.names";

                    var classes = File.ReadAllLines(classNamesFile).ToList();

                    Net net = CvDnn.ReadNetFromDarknet(modelConfig, modelWeights);

                    capture = new VideoCapture(videoPath);

                    if (!capture.IsOpened())
                    {
                        MessageBox.Show("Error to open video");
                        return;
                    }

                    Mat frame = new Mat();
                    int frameSkip = 2; // prossess in frame
                    int frameCounter = 0;
                    isVideoPlaying = true; 

                    while (!cancellationToken.IsCancellationRequested && isVideoPlaying)
                    {
                        if (!capture.Read(frame))
                        {
                            Dispatcher.Invoke(() =>
                            {
                                MessageBox.Show("Video ai detection operation completed. Please select a new video.");
                            });
                            break;
                        }

                        if (frame.Empty())
                            break;

                        frameCounter++;
                        if (frameCounter % frameSkip != 0)
                            continue;

                        Mat resizedFrame = new Mat();
                        Cv2.Resize(frame, resizedFrame, new OpenCvSharp.Size(640, 360));

                        Mat blob = CvDnn.BlobFromImage(resizedFrame, 1 / 255.0, new OpenCvSharp.Size(416, 416), new Scalar(0, 0, 0), true, false);
                        net.SetInput(blob);

                        var outputLayerNames = net.GetUnconnectedOutLayersNames();
                        var outputs = outputLayerNames.Select(_ => new Mat()).ToList();
                        net.Forward(outputs, outputLayerNames);

                        foreach (var output in outputs)
                        {
                            for (int i = 0; i < output.Rows; i++)
                            {
                                var scores = output.Row(i).ColRange(5, output.Cols);
                                Cv2.MinMaxLoc(scores, out _, out double maxVal, out _, out OpenCvSharp.Point classIdPoint);
                                int classId = classIdPoint.X;
                                double confidence = maxVal;

                                if (confidence > 0.2)
                                {
                                    string label = classes[classId];
                                    if (label == "person" || label == "cat" || label == "dog" || label == "horse" || label == "bird")
                                    {
                                        float centerX = output.At<float>(i, 0) * resizedFrame.Width;
                                        float centerY = output.At<float>(i, 1) * resizedFrame.Height;
                                        float width = output.At<float>(i, 2) * resizedFrame.Width;
                                        float height = output.At<float>(i, 3) * resizedFrame.Height;
                                        int x = (int)(centerX - width / 2);
                                        int y = (int)(centerY - height / 2);

                                        // رسم مستطیل
                                        Cv2.Rectangle(resizedFrame, new OpenCvSharp.Rect(x, y, (int)width, (int)height), Scalar.Red, 2);
                                        string text = $"{label} {confidence * 100:0}%";
                                        Cv2.PutText(resizedFrame, text, new OpenCvSharp.Point(x, y - 10), HersheyFonts.HersheySimplex, 0.2, Scalar.Green, 1);
                                    }
                                }
                            }
                        }

                        Dispatcher.Invoke(() =>
                        {
                            var bitmap = MatToBitmap(resizedFrame);
                            VideoImage.Source = BitmapToImageSource(bitmap);
                        });

                        if (Cv2.WaitKey(30) == 'q')
                            break;
                    }

                    capture.Release();
                    isVideoPlaying = false;

                }
                catch (Exception ex)
                {
                    Dispatcher.Invoke(() =>
                    {
                        MessageBox.Show($"خطا: {ex.Message}");
                    });
                }
            }, cancellationToken);
        }

        private void StopCurrentVideoProcessing()
        {
            if (videoProcessingTask != null && !videoProcessingTask.IsCompleted)
            {
                cancellationTokenSource?.Cancel();
                videoProcessingTask.Wait();
                capture?.Release();  
            }
        }


        private System.Drawing.Bitmap MatToBitmap(Mat mat)
        {
            if (mat.Empty())
                return null;

            byte[] imageData = mat.ToBytes();
            using (var ms = new MemoryStream(imageData))
            {
                return new System.Drawing.Bitmap(ms);
            }
        }
        private System.Windows.Media.ImageSource BitmapToImageSource(System.Drawing.Bitmap bitmap)
        {
            var hbitmap = bitmap.GetHbitmap();
            return System.Windows.Interop.Imaging.CreateBitmapSourceFromHBitmap(hbitmap, IntPtr.Zero, System.Windows.Int32Rect.Empty, System.Windows.Media.Imaging.BitmapSizeOptions.FromEmptyOptions());
        }

        private void Window_Drop(object sender, DragEventArgs e)
        {
            var droppedFiles = e.Data.GetData(DataFormats.FileDrop) as string[];
            if (droppedFiles != null && droppedFiles.Length > 0)
            {
                videoPath = droppedFiles[0];
                StartObjectDetection(videoPath);
            }
        }

        private void Window_DragOver(object sender, DragEventArgs e)
        {
            if (e.Data.GetDataPresent(DataFormats.FileDrop))
            {
                e.Effects = DragDropEffects.Copy;
            }
        }
    }
}

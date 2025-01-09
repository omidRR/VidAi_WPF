using Microsoft.Win32;
using OpenCvSharp;
using OpenCvSharp.Dnn;
using System.IO;
using System.Windows;
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

        private void ReplayButton_Click(object sender, RoutedEventArgs e)
        {
            if (!string.IsNullOrEmpty(videoPath))
            {
                StopVideoProcessingCompletely(); 
                StartObjectDetection(videoPath); 
            }
            else
            {
                MessageBox.Show("No video selected. Please select a video first.");
            }
        }

        private void StartObjectDetection(string videoPath)
        {
            StopVideoProcessingCompletely(); 

            cancellationTokenSource = new CancellationTokenSource();
            var cancellationToken = cancellationTokenSource.Token;

            videoProcessingTask = Task.Run(() =>
            {
                try
                {
                    string basePath = AppDomain.CurrentDomain.BaseDirectory;
                    string dataFolder = Path.Combine(basePath, "Data");

                    string modelWeights = Path.Combine(dataFolder, "yolov3-tiny.weights");
                    string modelConfig = Path.Combine(dataFolder, "yolov3-tiny.cfg");
                    string classNamesFile = Path.Combine(dataFolder, "coco.names");

                    var classes = File.ReadAllLines(classNamesFile).ToList();

                    Net net = CvDnn.ReadNetFromDarknet(modelConfig, modelWeights);
                    net.SetPreferableBackend(Backend.OPENCV);
                    net.SetPreferableTarget(Target.OPENCL);

                    capture = new VideoCapture(videoPath); 

                    if (!capture.IsOpened())
                    {
                        Dispatcher.Invoke(() => MessageBox.Show("Error to open video"));
                        return;
                    }

                    Mat frame = new Mat();
                    int frameSkip = 3; // process in 1 frame
                    int frameCounter = 0;
                    isVideoPlaying = true;

                    while (!cancellationToken.IsCancellationRequested && isVideoPlaying)
                    {
                        if (!capture.Read(frame) || frame.Empty())
                        {
                            Dispatcher.Invoke(() => MessageBox.Show("Video AI detection operation completed. Please select a new video."));
                            break;
                        }

                        frameCounter++;
                        if (frameCounter % frameSkip != 0)
                            continue;

                        Mat resizedFrame = new Mat();
                        Cv2.Resize(frame, resizedFrame, new OpenCvSharp.Size(900, 620));

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

                                    bool detectHumansOnly = Dispatcher.Invoke(() => HumanDetectionCheckBox.IsChecked == true);
                                    bool detectAllObjects = Dispatcher.Invoke(() => AllObjectsDetectionCheckBox.IsChecked == true);

                                    if (detectAllObjects && label != "person")
                                    {
                                        DrawDetection(resizedFrame, output, i, label, confidence);
                                    }
                                    else if (detectHumansOnly && label == "person")
                                    {
                                        DrawDetection(resizedFrame, output, i, label, confidence);
                                    }
                                }
                            }
                        }

                        Dispatcher.Invoke(() =>
                        {
                            var bitmap = MatToBitmap(resizedFrame);
                            VideoImage.Source = BitmapToImageSource(bitmap);
                        });

                        Thread.Sleep(15);
                    }

                    capture.Release();
                    isVideoPlaying = false;
                }
                catch (Exception ex)
                {
                    Dispatcher.Invoke(() => MessageBox.Show($"خطا: {ex.Message}"));
                }
            }, cancellationToken);
        }

     
        private void StopVideoProcessingCompletely()
        {
            if (videoProcessingTask != null && !videoProcessingTask.IsCompleted)
            {
                cancellationTokenSource?.Cancel(); 
                try
                {
                    videoProcessingTask.Wait(TimeSpan.FromSeconds(2)); 
                }
                catch (AggregateException)
                {
                   
                }
                finally
                {
                    capture?.Release(); 
                    isVideoPlaying = false; 
                    cancellationTokenSource?.Dispose(); 
                    cancellationTokenSource = null; 
                }
            }
        }

        private void DrawDetection(Mat frame, Mat output, int index, string label, double confidence)
        {
            float centerX = output.At<float>(index, 0) * frame.Width;
            float centerY = output.At<float>(index, 1) * frame.Height;
            float width = output.At<float>(index, 2) * frame.Width;
            float height = output.At<float>(index, 3) * frame.Height;
            int x = (int)(centerX - width / 2);
            int y = (int)(centerY - height / 2);

            Cv2.Rectangle(frame, new OpenCvSharp.Rect(x, y, (int)width, (int)height), Scalar.DarkBlue, 2);
            string text = $"{label} {confidence * 100:0}%";
            Cv2.PutText(frame, text, new OpenCvSharp.Point(x, y - 10), HersheyFonts.HersheySimplex, 0.5, Scalar.Green, 1);
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
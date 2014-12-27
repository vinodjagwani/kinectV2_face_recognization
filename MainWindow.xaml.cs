using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using Emgu.CV.Structure;
using Emgu.CV;
using System.Runtime.InteropServices;
using System.Windows.Threading;
using Microsoft.Kinect;
using System.IO;
using System.Threading;
using System.Drawing.Imaging;
using System.Xml;
using System.Diagnostics;
using Emgu.CV.GPU;
using System.Data;
using System.Data.SqlClient;

namespace KinectfaceProject
{



    public partial class MainWindow : Window
    {

        private CascadeClassifier haarCascade;
        DispatcherTimer timer;
        static string path = Environment.GetFolderPath(Environment.SpecialFolder.Desktop);
        ClassifierTrain EigenRecognition = new ClassifierTrain(path+"/FaceDB/");
        XmlDocument docu = new XmlDocument();
        private bool doBody = true;
        private WriteableBitmap bitmap = null;
        Image<Bgra, Byte> currentColorImageFrame;
        private readonly int bytesPerPixel = (PixelFormats.Bgr32.BitsPerPixel + 7) / 8;
        private KinectSensor kinectSensor = null;
        private MultiSourceFrameReader reader = null;
        private byte[] pixels = null;
        private Body[] bodies = null;
        private CoordinateMapper coordinateMapper = null;
        ParallelOptions po = new ParallelOptions();
        List<ulong> myList = new List<ulong>();

        public ImageSource ImageSource
        {
            get
            {
                return this.bitmap;
            }
        }

        public string StatusText
        {
            get { return (string)GetValue(StatusTextProperty); }
            set { SetValue(StatusTextProperty, value); }
        }

        // Using a DependencyProperty as the backing store for StatusText.  This enables animation, styling, binding, etc...
        public static readonly DependencyProperty StatusTextProperty =
            DependencyProperty.Register("StatusText", typeof(string), typeof(MainWindow), new PropertyMetadata(""));

        public int EigenDistance
        {
            get { return (int)GetValue(EigenDistanceProperty); }
            set { SetValue(EigenDistanceProperty, value); }
        }

        // Using a DependencyProperty as the backing store for EigenDistance.  This enables animation, styling, binding, etc...
        public static readonly DependencyProperty EigenDistanceProperty =
            DependencyProperty.Register("EigenDistance", typeof(int), typeof(MainWindow), new PropertyMetadata(0));



        public MainWindow()
        {

            this.kinectSensor = KinectSensor.Default;
            if (this.kinectSensor != null)
            {
                this.coordinateMapper = this.kinectSensor.CoordinateMapper;
                this.kinectSensor.Open();
                FrameDescription ColorframeDescription = this.kinectSensor.ColorFrameSource.FrameDescription;
                this.reader = this.kinectSensor.OpenMultiSourceFrameReader(FrameSourceTypes.Color | FrameSourceTypes.Body);
                this.bodies = new Body[this.kinectSensor.BodyFrameSource.BodyCount];
                this.pixels = new byte[ColorframeDescription.Width * ColorframeDescription.Height * this.bytesPerPixel];
                this.bitmap = new WriteableBitmap(ColorframeDescription.Width, ColorframeDescription.Height, 96.0, 96.0, PixelFormats.Bgr32, null);
            }
            this.DataContext = this;

            if (EigenRecognition.IsTrained)
            {
                StatusText = "Detect";
            }
            else
            {
                StatusText = "Not detect";
            }
            this.Loaded += MainWindow_Loaded;
            this.Closing += MainWindow_Closing;
            InitializeComponent();
        }


        void MainWindow_Closing(object sender, System.ComponentModel.CancelEventArgs e)
        {
            if (this.reader != null)
            {

                this.reader.Dispose();
                this.reader = null;
            }


            if (this.bodies != null)
            {
                foreach (Body body in this.bodies)
                {
                    if (body != null)
                    {
                        body.Dispose();
                    }
                }
            }

            if (this.kinectSensor != null)
            {
                this.kinectSensor.Close();
                this.kinectSensor = null;
            }


        }

        void MainWindow_Loaded(object sender, RoutedEventArgs e)
        {

            haarCascade = new CascadeClassifier(@"haarcascade_frontalface_default.xml");
            timer = new DispatcherTimer();
            timer.Tick += new EventHandler(timer_Tick);
            timer.Interval = new TimeSpan(0, 0, 0, 0, 30);
            EigenRecognition.Set_Eigen_Threshold = 2000;
            po.MaxDegreeOfParallelism = Environment.ProcessorCount;
            if (this.reader != null)
            {
                this.reader.MultiSourceFrameArrived += reader_MultiSourceFrameArrived;
            }
            timer.Start();

          }

        private void reader_MultiSourceFrameArrived(object sender, MultiSourceFrameArrivedEventArgs e)
        {
            MultiSourceFrameReference frameReference = e.FrameReference;
            MultiSourceFrame multiSourceFrame = null;
            ColorFrame colorFrame = null;
            BodyFrame bodyFrame = null;

            try
            {
                multiSourceFrame = frameReference.AcquireFrame();
               
                if (multiSourceFrame != null)
                {
                    using (multiSourceFrame)
                    {
                        ColorFrameReference colorFrameReference = multiSourceFrame.ColorFrameReference;
                        BodyFrameReference bodyFrameReference = multiSourceFrame.BodyFrameReference;

                        colorFrame = colorFrameReference.AcquireFrame();
                        bodyFrame = bodyFrameReference.AcquireFrame();
                        if (colorFrame != null && bodyFrame != null)
                        {


                            FrameDescription colorFrameDescription = colorFrame.FrameDescription;

                            if ((colorFrameDescription.Width == this.bitmap.PixelWidth) && (colorFrameDescription.Height == this.bitmap.PixelHeight))
                            {

                                if (colorFrame.RawColorImageFormat == ColorImageFormat.Bgra)
                                {
                                    colorFrame.CopyRawFrameDataToArray(this.pixels);
                                }
                                else
                                {
                                    colorFrame.CopyConvertedFrameDataToArray(this.pixels, ColorImageFormat.Bgra);
                                }
                                System.Drawing.Bitmap bitmapColorImageFrame = new System.Drawing.Bitmap(colorFrameDescription.Width, colorFrameDescription.Height, System.Drawing.Imaging.PixelFormat.Format32bppRgb);
                                System.Drawing.Imaging.BitmapData bitmapDataColorImageFrame = bitmapColorImageFrame.LockBits(new System.Drawing.Rectangle(0, 0, colorFrameDescription.Width, colorFrameDescription.Height), System.Drawing.Imaging.ImageLockMode.WriteOnly, bitmapColorImageFrame.PixelFormat);
                                IntPtr ptr = bitmapDataColorImageFrame.Scan0;
                                Marshal.Copy(this.pixels, 0, ptr, colorFrameDescription.Width * colorFrameDescription.Height * bytesPerPixel);
                                bitmapColorImageFrame.UnlockBits(bitmapDataColorImageFrame);
                                currentColorImageFrame = new Image<Bgra, Byte>(bitmapColorImageFrame);
                                this.bitmap.WritePixels(new Int32Rect(0, 0, colorFrameDescription.Width, colorFrameDescription.Height), this.pixels, colorFrameDescription.Width * this.bytesPerPixel, 0);
                            }

                            if (doBody)
                            {
                                bodyFrame.GetAndRefreshBodyData(this.bodies);
                                canvas.Children.Clear();
                                foreach (Body body in this.bodies)
                                {
                                   
                                    if (body.IsTracked)
                                    {
                                        IReadOnlyDictionary<JointType, Joint> joints = body.Joints;
                                        Dictionary<JointType, Point> jointPoints = new Dictionary<JointType, Point>();
                                        ColorSpacePoint colorSpacePoint = this.coordinateMapper.MapCameraPointToColorSpace(joints[JointType.Head].Position);
                                        double userDistance = joints[JointType.Neck].Position.Z;
                                        if (userDistance >= 1.7 && userDistance <= 1.9)
                                        {
                                        Rectangle myx = new Rectangle();
                                        myx.Stroke = Brushes.Red;
                                        myx.StrokeThickness = 4;
                                        myx.Height = (int)260 / userDistance;
                                        myx.Width = (int)220 / userDistance;
                                        Console.WriteLine(userDistance);
                                        canvas.Children.Add(myx);
                                        int Xoffset = (int)(100 / userDistance);
                                        int Yoffset = (int)(90 / userDistance);
                                        int positionX = (int)colorSpacePoint.X - Xoffset;
                                        int positionY = (int)colorSpacePoint.Y - Yoffset;
                                        Canvas.SetLeft(myx, positionX);
                                        Canvas.SetTop(myx, positionY);
                                        
                                        Image<Gray, byte> face = currentColorImageFrame.Copy(new System.Drawing.Rectangle(positionX, positionY, (int)myx.Width, (int)myx.Height)).Convert<Gray, byte>().Resize(100, 100, Emgu.CV.CvEnum.INTER.CV_INTER_CUBIC);
                                        face._EqualizeHist();
                                        
                                        if (EigenRecognition.IsTrained)
                                        {
                                                string name = EigenRecognition.Recognise(face);
                                                int match_value = (int)EigenRecognition.Get_Eigen_Distance;
                                                TextBlock detectedUser = new TextBlock() { Text = name, FontSize = 40, Foreground = Brushes.Blue };
                                                canvas.Children.Add(detectedUser);
                                                Canvas.SetLeft(detectedUser, positionX);
                                                Canvas.SetTop(detectedUser, positionY);
                                                if (name.Trim().Length > 0 && !"Unknown".Equals(name.Trim()))
                                                {
                                                    
                                                    if (DateTime.Now.Hour >= 8 && DateTime.Now.Hour <= 11)
                                                    {
                                                        if (check_InEmployee(name.Trim(), System.DateTime.Today.ToShortDateString()) == false)
                                                        {
                                                            checkIn(name.Trim());
                                                        }
                                                    }
                                                    else if (DateTime.Now.Hour >= 16 && DateTime.Now.Hour <= 18)
                                                    {
                                                        if (check_OutEmployee(name.Trim(), System.DateTime.Today.ToShortDateString()) == false)
                                                        {
                                                            checkOut(name.Trim());
                                                        }
                                                     }
                                                 }
                                            }
                                            face.Dispose();
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
            }
            finally
            {
                if (colorFrame != null)
                {
                    colorFrame.Dispose();
                    colorFrame = null;
                }
                if (bodyFrame != null)
                {
                    bodyFrame.Dispose();
                    bodyFrame = null;
                }
            }
        }





        void timer_Tick(object sender, EventArgs e)
        {

            // refresh the frames
            if (this.reader != null)
            {
                this.reader.MultiSourceFrameArrived += reader_MultiSourceFrameArrived;

            }

        }

        
       
        private void Button_Close(object sender, RoutedEventArgs e)
        {
            this.Close();
        }


        private void checkIn(string name)
        {
            int check = 0;
            SqlConnection sqlCnn = DBConnection.getConnection();
            try
            {
                sqlCnn.Open();
                string cmd = "insert into [dbo].[Table] (name,checkin,inouttime) values('" + name + "',1,'" + System.DateTime.Today.ToShortDateString() + "')";
                SqlCommand sqlcmd = new SqlCommand(cmd, sqlCnn);
                check = sqlcmd.ExecuteNonQuery();
                if (check == 1)
                {
                    MessageBox.Show(name +" Check in Successfully!");
                }
                
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message);
            }
            finally
            {
                sqlCnn.Close();
            }
            
        }

        private void checkOut(string name)
        {
            int check = 0;
            SqlConnection sqlCnn = DBConnection.getConnection();
            try
            {
                sqlCnn.Open();
                string cmd = "update[dbo].[Table] set checkout=1,inouttime='" + System.DateTime.Today.ToShortDateString() + "' where name='" + name + "' and checkin=1";
                SqlCommand sqlcmd = new SqlCommand(cmd, sqlCnn);
                check = sqlcmd.ExecuteNonQuery();
                if (check == 1)
                {
                    MessageBox.Show(name + " Check out Successfully!");
                }
            }
            catch (Exception ex)
            {
                
                MessageBox.Show(ex.Message);
            }
            finally
            {
                sqlCnn.Close();
            }
            
        }

        private bool check_InEmployee(string name, string datetime)
        {

            bool checkEmployeeDB = false;
            SqlConnection sqlCnn = DBConnection.getConnection();
            SqlDataReader dReader;
            SqlCommand cmd = null;
            try
            {
                string query = "select * from [dbo].[Table] where name='" + name + "' and checkin = 1   and inouttime='" + datetime + "' ";
                cmd = new SqlCommand();
                cmd.Connection = sqlCnn;
                cmd.CommandType = CommandType.Text;
                cmd.CommandText = query;
                sqlCnn.Open();
                dReader = cmd.ExecuteReader();
                if (dReader.HasRows == true)
                {
                    checkEmployeeDB = true;
                }
                else
                {
                    return checkEmployeeDB;
                }
                dReader.Close();
                return checkEmployeeDB;
            }
            catch (Exception ex)
            {
                sqlCnn.Close();
                MessageBox.Show(ex.Message);
            }
            finally
            {
                sqlCnn.Close();
            }
            return checkEmployeeDB;
        }


        private bool check_OutEmployee(string name, string datetime)
        {

            bool checkEmployeeDB = false;
            SqlConnection sqlCnn = DBConnection.getConnection();
            SqlDataReader dReader;
            SqlCommand cmd = null;
            try
            {
                string query = "select * from [dbo].[Table] where name='" + name + "' and checkin = 1 and checkout = 1 and inouttime='" + datetime + "' ";
                cmd = new SqlCommand();
                cmd.Connection = sqlCnn;
                cmd.CommandType = CommandType.Text;
                cmd.CommandText = query;
                sqlCnn.Open();
                dReader = cmd.ExecuteReader();
                if (dReader.HasRows == true)
                {
                    checkEmployeeDB = true;
                }
                else
                {
                    return checkEmployeeDB;
                }
                dReader.Close();
                return checkEmployeeDB;
            }
            catch (Exception ex)
            {
                sqlCnn.Close();
                MessageBox.Show(ex.Message);
            }
            finally
            {
                sqlCnn.Close();
            }
            return checkEmployeeDB;
        }


    }
}
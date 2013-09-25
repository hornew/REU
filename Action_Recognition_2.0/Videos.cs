/********************************************************************************************************************
 * File Name            : Videos.cs
 * Purpose              : Handles all input of video clips, frame separation and scaling, and feature extraction.
 *                          
 * Author               : Austin Horne     E-mail: hornew@goldmail.etsu.edu       
 * Date                 : July 2013
 * Modified by       : 
*********************************************************************************************************************
*/

using User.DirectShow;      //for use of FrameGrabber object. DirectShow is a 3rd-party library for working with media. FrameGrabber is an easy to use interface to the MediaDet object in DirectShow
                            //http://msdn.microsoft.com/en-us/library/windows/desktop/dd375454(v=vs.85).aspx

using weka.core.matrix;     //the Weka API is a 3rd-party library written in Java with various machine learning algorithms and utilities. http://weka.sourceforge.net/doc.dev/
                            //Using IKVM to interface between Java and C#/.NET   http://www.ikvm.net/   See documentation for more details.

using System.Drawing;       //for use of Bitmap and Color objects
using System.IO;            //for use of Directory class
using System;               //for use of Random and Console objects
using System.Collections.Generic;   //for List

namespace Action_Recognition_2._0
{
    class FeatureExtractor2D
    {
        public int SpatialSize { get; set; }
        public int TemporalSize { get; set; }
        public int NumPatches { get; set; }        
        public string[] TrainFiles { get; set; }


        public FeatureExtractor2D()
        {
        }

        public FeatureExtractor2D(string path, int spSize, int tmpSize, int patches)
        {
            SetProperties(spSize, tmpSize, patches);
        }

        public void ExtractTrainingData(string vidDir, string vidFileString, bool allFiles)
        {
            int margin = 5;
            int counter = 0;
            Random r = new Random();

            if (!allFiles)  //search for files matching a specific pattern, i.e. actioncliptrain*
            {
                vidFileString = vidFileString + "*";    //concatenate this into a search pattern so that get files returns only files that match 
                TrainFiles = Directory.GetFiles(vidDir, vidFileString);    //get all the files beginning with vidFileString and anything following. Returns a string[]
            }
            else
                TrainFiles = Directory.GetFiles(vidDir);    //get all files in the specified directory. Returns a string[]
    
            Matrix X = new Matrix(SpatialSize * SpatialSize * TemporalSize, TrainFiles.Length * NumPatches);

            for (int i = 0; i < TrainFiles.Length; i++)     //load in each clip from the list of videos and perform convolution
            {
                Console.WriteLine("loading clip: {0}", TrainFiles[i]);
                Matrix[] M = LoadClip(TrainFiles[i]);
                int xDim = M[0].getColumnDimension();   //number of columns = x dimension
                int yDim = M[0].getRowDimension();      //number of rows = y dimension
                int tDim = M.Length;                    //number of 2D matrices in M = t dimension

                for (int j = 0; j < NumPatches; j++)    //convolution step
                {
                    int lowerBound = ++margin;
                    int upperBound = xDim - margin - SpatialSize + 1;
                    int xPos = r.Next(lowerBound, upperBound);          //unlabeled data, so pick a random x.......

                    upperBound = yDim - margin - SpatialSize + 1;
                    int yPos = r.Next(lowerBound, upperBound);          //....and a random y....

                    upperBound = tDim - TemporalSize + 1;
                    int tPos = r.Next(1, upperBound);                   //....and a random t within the image boundaries from which to extract features

                    Matrix[] block = GetBlock(M, xPos, yPos, tPos);
                    double[] reshapedBlock = Reshape(block, false);
                    SetColumn(X, counter, reshapedBlock);
                    counter++;
                }
            }
                        
        }//ExtractTrainingData()


        /// <summary>
        /// Open the specified video and get the individual frames.
        /// Construct a 3D Matrix from the 2D matrices returned by the FixFrame function.
        /// </summary>
        /// <param name="path"></param>
        /// <returns></returns>
        private Matrix[] LoadClip(string path)
        {
            FrameGrabber frames = new FrameGrabber(path);   //grab frames from video at the specified path
            Matrix[] M;                        //Array of 2D matrices built from the frames in the video loaded. Essentially a 3D matrix
            int count = frames.FrameCount;     //number of frames in the video

            Matrix fixedFrame = FixFrame((Bitmap)frames.GetFrame(0)); //fix the first frame. Used for initialization of M

            M = new Matrix[count];

            for (int i = 0; i < count; i++)     //call FixFrame on each frame in the video and fill the matrix with these frames
            {
                fixedFrame = FixFrame((Bitmap)frames.GetFrame(i));                
                M[i] = fixedFrame;                
            }
            return M;    
        }//LoadClip()


        /// <summary>
        /// 
        /// </summary>
        /// <param name="M"></param>
        /// <param name="xPos"></param>
        /// <param name="yPos"></param>
        /// <param name="tPos"></param>
        /// <returns></returns>
        private Matrix[] GetBlock(Matrix[] M, int xPos, int yPos, int tPos)
        {
            int xUpperBound = xPos + SpatialSize - 1;
            int yUpperBound = yPos + SpatialSize - 1;
            int tUpperBound = tPos + TemporalSize - 1;
            Matrix[] Block = new Matrix[TemporalSize - 1];

            for (int i = 0, j = tPos; j < tUpperBound; i++, j++)
            {
                Block[i] = M[j].getMatrix(xPos, xUpperBound, yPos, yUpperBound);
            }
            return Block;
        }//GetBlock()

        /// <summary>
        /// Grayscale and crop frames
        /// </summary>
        /// <param name="map"></param>
        /// <returns></returns>
        private Matrix FixFrame(Bitmap map)
        {            
            int scalar = 1 / 255;
            int rowBound = (int)Math.Floor((double)(map.Width/SpatialSize)) * SpatialSize;
            int colBound = (int)Math.Floor((double)(map.Height/SpatialSize)) * SpatialSize;
            Matrix M;

            map = rgb2gray(map);
            M = ToMatrix(map);
            M = M.timesEquals(scalar);      //divide each value in the Matrix by 255. Note that scalar has a value of 1/255
            M = M.getMatrix(1, rowBound, 1, colBound);
            return M;
        }//FixFrame()

        /// <summary>
        /// Row-wise iteration through the Bitmap, get the color from each pixel and change to grayscale.
        /// Refer to this link for explanation of conversion http://stackoverflow.com/questions/687261/converting-rgb-to-grayscale-intensity
        /// 
        /// NOTE: In the future, consider converting Bitmap to Matrix here to set pixel value to integer instead of Color, essentially
        /// skipping the Color.FromArgb step
        /// </summary>
        /// <param name="bm"></param>
        /// <returns></returns>
        public Bitmap rgb2gray(Bitmap bm)
        {
            //Row-wise iteration through the Bitmap 
            for (int y = 0; y < bm.Height; y++)
            {
                for (int x = 0; x < bm.Width; x++)
                {
                    Color pixelColor = bm.GetPixel(x, y);

                    int pixelLuminance = (int)(pixelColor.R * 0.2126 + pixelColor.G * 0.7152 + pixelColor.B * 0.0722);

                    bm.SetPixel(x, y, Color.FromArgb(pixelLuminance, pixelLuminance, pixelLuminance));
                }//for
            }//for

            return bm;
        }//rgb2gray(Bitmap)

        /// <summary>
        /// Copies the data from a Bitmap object into a Weka Matrix for purposes of matrix manipulation.
        /// Future consideration:  Bitmap class's LockBits() function offers better performance for large-scale changes that SetPixel()
        /// </summary>
        /// <returns></returns>
        public Matrix ToMatrix(Bitmap bm)
        {
            int height = bm.Height;
            int width = bm.Width;
            Matrix M = new Matrix(height, width);
            Color pixelColor;

            //iterate through the Bitmap, getting the colors from each pixel, convert the colors to RGB alpha value and store in a Matrix
            for (int i = 0; i < width; i++)
            {
                for (int j = 0; j < height; j++)
                {
                    pixelColor = bm.GetPixel(i, j);     //get pixel color from Bitmap
                    M.set(i, j, pixelColor.ToArgb());   //convert to alpha RGB and store in the Matrix, M
                }
            }

            return M;
        }//ToMatrix(Bitmap)

        /// <summary>
        /// Function to change SpatialSize, TemporalSize, and NumPatches with a single call for the sake of convenience
        /// </summary>
        /// <param name="spSize"></param>
        /// <param name="tmpSize"></param>
        /// <param name="patches"></param>
        public void SetProperties(int spSize, int tmpSize, int patches)
        {
            this.SpatialSize = spSize;
            this.TemporalSize = tmpSize;
            this.NumPatches = patches;
        }//SetProperties()

        /// <summary>
        /// Takes a 3D Matrix (2D Matrix array) and copies the data into a single dimension vector (double array).
        /// Can be specified to copy data row - wise or column -wise from the input Matrix using the second parameter
        /// Functions differently from the Matlab version of Reshape in that this version assumes that the input Matrix
        /// will be reshaped into a single dimension.  This functionality works for our purposes.
        /// </summary>
        /// <param name="block"></param>
        /// <param name="rowPacked"></param>
        /// <returns>reshapedBlock</returns>
        private double[] Reshape(Matrix[] block, bool rowPacked)
        {            
            int xDim = block[0].getColumnDimension();       //x and y dimensions are the same for all items in the block Matrix
            int yDim = block[0].getRowDimension();
            int tDim = block.Length;
            List<double> reshapedBlock = new List<double>(xDim * yDim * tDim);  //initialized to the number of elements in the 3D matrix for optimization

            if (rowPacked)  //builds an array from row packed copy of block
            {
                foreach (var item in block)
                    reshapedBlock.AddRange(item.getRowPackedCopy());                
            }
            else            //builds an array from column packed copy of block
            {
                foreach (var item in block)
                    reshapedBlock.AddRange(item.getColumnPackedCopy());
            }

            return reshapedBlock.ToArray();
        }//Reshape()

        /// <summary>
        /// Sets a column of Matrix m specified by the index col to the values contained in the double array vals.
        /// Row dimension of Matrix m must match the length of array vals. Throws an exception if these dimensions do not match
        /// </summary>
        /// <param name="m"></param>
        /// <param name="col"></param>
        /// <param name="vals"></param>
        private void SetColumn(Matrix m, int col, double[] vals)
        { 
            if(m.getRowDimension() != vals.Length)  //disallow inconsistent dimensions
                throw new InconsistentDimensionException("Dimension mismatch: Array length and Matrix row dimension must be equal");

            for (int i = 0; i < m.getRowDimension(); i++)
                m.set(i, col, vals[i]);
        }//SetColumn()
        
    }//class
}//namespace

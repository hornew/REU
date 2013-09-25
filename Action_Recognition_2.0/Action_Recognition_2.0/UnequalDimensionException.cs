/********************************************************************************************************************
 * File Name            : Videos.cs
 * Purpose              : Handles all input of video clips, frame separation and scaling, and feature extraction.
 *                          
 * Author               : Austin Horne     E-mail: hornew@goldmail.etsu.edu       
 * Date                 : July 2013
 * Modified by       : 
*********************************************************************************************************************
*/
using System;
namespace Action_Recognition_2._0
{
    class InconsistentDimensionException : Exception
    {
        public InconsistentDimensionException(string err) : base(err)
        { }
    }//class
}//namespace

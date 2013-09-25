using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using User.DirectShow;
using weka.core.matrix;

namespace Action_Recognition_2._0
{
    class Program
    {
        static void Main(string[] args)
        {
            Matrix M = new Matrix(10, 10);

            for (int i = 0; i < M.getRowDimension(); i++)
            {
                for (int j = 0; j < M.getColumnDimension(); j++)
                {
                    M.set(j, i, ((i + 1) * j));
                }
            }

            
            //M.print(...);
            /*double[] d = new double[25];

            for (int i = 0; i < 25; i++)
			{
			    d[i] = i + 1;
			}

            Matrix mat = new Matrix(d, 5);

            for (int i = 0; i < mat.getColumnDimension(); i++)
            {
                for (int j = 0; j < mat.getRowDimension(); j++)
                {
                    Console.Write(mat.get(j,i) + " ");
                }
                Console.WriteLine();
            }*/
        }//Main
    }
}

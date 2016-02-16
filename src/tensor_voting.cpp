/*
 * =====================================================================================
 *
 *       Filename:  tensor_voting.cpp
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  03/30/2012 04:27:18 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Ming Liu (), ming.liu@mavt.ethz.ch
 *        Company:  ETHZ
 *
 * =====================================================================================
 */

#include "tensor_voting.h"
#include <assert.h>
#include <iostream>

#include <fstream>

#define do_sparse_voting_first


using namespace std;
namespace topomap{

    void TensorVoting::setPoints( const DP & pointsVector)
    {
        unsigned int cols = pointsVector.features.cols();
        points.resize(3, cols);
        points = pointsVector.features.topRows(3);
    }

    void TensorVoting::setPoints( const LinkedObstacleType & pointsVector)
    {
        unsigned int cols = pointsVector.size();
        points.resize(3, cols);
#ifdef use_PoseID
        for( unsigned int i=0; i<pointsVector.size(); i++) // each cellPoseID in this obstacle
        {
            PoseIDType p = pointsVector[i];
            points.col(i) << p.get<0>(), p.get<1>(), p.get<2>();
        }
#else
        for( unsigned int i=0; i<pointsVector.size(); i++) // each cellPoseID in this obstacle
        {
            PoseIDType p = pointsVector[i];
            points.col(i) << float(p.get<0>())/1000.0, float(p.get<1>())/1000.0, float(p.get<2>())/1000.0;
        }

#endif
        assert( points.cols() == pointsVector.size() );
     //   cout << "Number of points: " << points.cols() << endl;
     //   cout << points << endl;
    }

    void TensorVoting::setSparseTensor(const Matrix<Matrix3f,Dynamic,1> _sparseTensor)
    {
        sparseTensors = _sparseTensor;
    }

    void TensorVoting::setNormals( const NormalsType & normalsVector)
    {
        unsigned int cols = normalsVector.size();
        normals.resize(3, cols);
        for( unsigned int i=0; i<normalsVector.size(); i++) // normals for each cell in this obstacle
        {
            NormalType p = normalsVector[i];
            normals.col(i) << p.get<0>(), p.get<1>(), p.get<2>();
        }
        assert( normals.cols() == normalsVector.size() );
     //   cout << "Number of normals: " << normals.cols() << endl;
     //   cout << normals << endl;
    }

    Matrix<Matrix3f, Dynamic, 1> TensorVoting::stickDenseVote()
    {
        unsigned int numPoints = points.cols();
        field.resize(numPoints, 1); // may cause segfault, then change to 3*numPoints
        
        RowVector3f vn;             // stick vote normal space basis vector
        RowVector3f O;              // location of the voter (in m)
        float stick_saliency;// saliency of voter
        RowVector3i P_grid;// location of the votee (in cell coordinates)
        RowVector3f P;              // location of the votee (in m)
        RowVector3i xyz;            // vector from O to P (in cell coordinates)
        RowVector3f v;              // vector from O to P (in m)
        RowVector3f vt;             // stick vote tangent space basis vector
        float vvn;                  // V*Vn (dot product, see King pg. 69)
        float vvt;                  // V*Vt (dot product, see King pg. 69)
        float l;                    // voter-votee distance (in m)
        float theta;                // angle between v and vt (in rad)
        RowVector3f vc;             // votee normal vector
        float scaled_dist;          // scale normalized voter-votee distance
        float DF;                   // decay function ie. vote weighting
        Matrix3f Sp;                // tensor vote
        int index;                  // index for cell's tensor

        map<unsigned int, vector<unsigned int> > who_receied_whos;

        for(unsigned int i=0; i<field.rows(); i++)
            field(i).setZero();
//        cout << "Points: " << points << endl;
//
        // log 
        size_t sizeLog = numPoints*sizeof(int2);
        h_log = (int2 *)malloc(sizeLog);
        bzero( h_log, sizeLog);
        printf ( "Start logging\n" );
        ofstream logf("log_number_of_whowhom_dense_stick_cpu.txt");
        // go through all points
        for (unsigned int i=0; i<numPoints; i++)
        {
            printf("dense stick process: %d (%d) %.2f%%\r", i, numPoints, 100*float(i)/numPoints); 
            fflush(stdout);
#ifndef do_sparse_voting_first
            vn << normals.col(i).transpose(); // row vector
            stick_saliency = 1.0; // TODO: not sure how to get it
    //        cout << "Using normals." << endl;
#else
            vn << sparseStick.col(i).tail(3).transpose();
            stick_saliency = sparseStick(0, i);
     //       cout << "Using sparse vote (stick)." << endl;
#endif

            if (vn.squaredNorm() > 0) {vn /= vn.norm(); }// normalize
            O = points.col(i).cast<float>(); // in unit of mm of float
            // cout << "vote:" << i <<endl;
            //

            /* this test is ok.
            h_log[i].x = (int)(100*stick_saliency);
            h_log[i].y = (int)(100*vn.norm());
            */

   //         cout << "Theta sets for " << i << " : ";
            // collect votes from other points
            for (unsigned int j=0; j<numPoints; j++)
            {
                if(j == i) continue;
                P = points.col(j).cast<float>();
                v = P - O;
                l = v.norm();
                scaled_dist = l/sigma;

                /* Check points. Make sure it is not cleaned before use.. */
                /*
                if (i==10)
                {
                        h_log[j].x = (int)(100*P(0));
                        h_log[j].y = (int)(100*P(1));
                }
                */
                /*  
                if (i==10)
                {
                        h_log[j].x = (int)(100*l);
                }
                if (i==1000)
                {
                        h_log[j].y = (int)(l*100);
                }
                */


                // only consider points reasonablly near
                if (scaled_dist<3 && l>0)
                {

                    v /= l; // normalize
                    vvn = v.dot(vn); // size of points
                    if (vvn<0) {vn *= -1;}

                    theta = asin(vvn); // separation angle O->P

//                    if(theta != 0 && theta!=M_PI/2){
//                        cout << i << "->" << j << endl;
//                        cout << "Theta: " << theta << endl;
//                        cout << "vn: " << vn << endl;
//                        cout << "v: " << v << endl;
//                        cout << "vvn: " << vvn << endl;
//                    }
                   
                    if (fabs(theta) <= M_PI/4 || fabs(theta) >= 3*M_PI/4)
                    {
                        vt = vn.cross(v.cross(vn));
                        if (vt.squaredNorm() > 0) { vt /= vt.norm(); } //normalize
                        vvt = v.dot(vt); // theta angle cosine





                        // TODO: evaluate again for 3d data
                        //cout << "Pt: " << i << " to " << j << " vvt: "<< vvt << " cos(theta)" << cos(theta) << endl;
                        if (vvt<0) { vt *= -1; } // [KING p.69]

                        vc = vn*cos(2*theta) - vt*sin(2*theta); //TODO: to be determined
                        // method 1
//                        DF = radial_decay_smooth(scaled_dist)*angular_decay(theta);
//                        Sp = DF*stick_saliency*(vc.transpose()*vc);

                        // PASSED: vt vvt
                        /*
                        if (i==10)
                        {
                            h_log[j].x = (int)(1000*vt.norm());
                            h_log[j].y = (int)(1000*vvt);
                        }
                        */

                        // PASSED: vc
                        /*
                           if (i==10)
                           {
                           h_log[j].x = (int)(1000*vc(0));
                           h_log[j].y = (int)(1000*vc(1));
                           }
                        */
                        // method 2
                        float c = radial_decay_smooth(scaled_dist);
                        float r = vc.norm();
                        DF=exp( -(4*r*r*theta*theta + c/(r*r)) / (sigma*sigma) );
                        Sp = stick_saliency*DF*(vc.transpose()*vc);
                        //Sp = DF*(vc.transpose()*vc);

                        // test of DF and c: PASSED
//                        if (i==10)
//                        {
//                            h_log[j].x = (int)(1000*c);
//                            h_log[j].y = (int)(1000*DF);
//                        }

                        //field(i) += Sp; // should be wrong
                        field(j) += Sp;
//                        cout << theta/M_PI  << ' '<< "DF" << DF << ' ' << "refcos: " << vvt << "cos: " << cos(theta) << endl;

                        // test of used Sp: PASSED
//                        if (i==10)
//                        {
//                            h_log[j].x = (int)(Sp(0,0)*1000);
//                            h_log[j].y = (int)(Sp(1,1)*1000);
//                        }

                        // test votee
                        if (j==66)
                        {
                            h_log[i].x = (int)(Sp(0,0)*1000);
                            h_log[i].y = (int)(Sp(1,1)*1000);
                        }
                        // update the log: who_receied_whos
                        if (who_receied_whos.find(j) != who_receied_whos.end())
                        { // there already
                            who_receied_whos[j].push_back(i);
                        }
                        else
                        { // create structure
                            vector<unsigned int> senders;
                            senders.reserve(numPoints);
                            senders.push_back(i);
                            who_receied_whos[j]=senders;
                        }

                     //   h_log[i].x += 1;
                     //   h_log[j].y += 1;


                    }
                }
            } // all neighbouring points
  //          cout << endl << "Tensor Element origined from " << i << " finished." << endl;

        } // all points to be processed

//        cout << "Who received whos message: " << endl;
//        for(map<unsigned int, vector<unsigned int> >::iterator iter = who_receied_whos.begin();
//                    iter !=who_receied_whos.end();
//                    ++ iter)
//        {
//            cout << iter->first << '\t' << iter->second.size()<< '\t';
//            for (unsigned int i = 0; i<iter->second.size(); i++)
//            {
//                cout << iter->second[i] << " ";
//            }
//            cout << endl;
//        }

        for(int i =0; i<numPoints; i++)
        {
            logf << i << " " << h_log[i].x << " " << h_log[i].y << endl;
        }
        free(h_log);
        logf.close();
        printf ("\n dense whotowhom log finished... \n");

        return field;
    } // end of function stickdensevote


    Matrix<Matrix3f, Dynamic, 1> TensorVoting::plateDenseVote()
    {
        unsigned int numPoints = points.cols();
        field.resize(numPoints, 1); // may cause segfault, then change to 3*numPoints
        
        RowVector3f vn;             // stick vote normal space basis vector
        RowVector3f O;              // location of the voter (in m)
        float plate_saliency;// saliency of voter
        RowVector3i P_grid;// location of the votee (in cell coordinates)
        RowVector3f P;              // location of the votee (in m)
        Matrix<float,3,2> U;		// matrix with first d eigenvalues as columns
        Matrix3f Np;				// voter's normal space
        RowVector3i xyz;            // vector from O to P (in cell coordinates)
        RowVector3f v;              // vector from O to P (in m)
        RowVector3f vt;             // stick vote tangent space basis vector
        float vvn;                  // V*Vn (dot product, see King pg. 69)
        float vvt;                  // V*Vt (dot product, see King pg. 69)
        float l;                    // voter-votee distance (in m)
        float theta;                // angle between v and vt (in rad)
        RowVector3f vc;             // votee normal vector
        float scaled_dist;          // scale normalized voter-votee distance
        float DF;                   // decay function ie. vote weighting
        Matrix3f Sp;                // tensor vote
        int index;                  // index for cell's tensor

        map<unsigned int, vector<unsigned int> > who_receied_whos;

        // ! Wierd but only way to work. Vote collection not starting from 0, but sparse tensors.
        // ! without this, the voting result is reversed. See line 524 of voronoi.cpp
        field = sparseTensors;

        for(unsigned int i=0; i<field.rows(); i++)
            field(i).setZero();

        cout << "Points: " << points << endl;


        // log 
        size_t sizeLog = numPoints*sizeof(int2);
        h_log = (int2 *)malloc(sizeLog);
        bzero( h_log, sizeLog);
        printf ( "Start logging\n" );
        ofstream logf("log_number_of_whowhom_dense_plate_cpu.txt");

        // go through all points, voters
        for (unsigned int i=0; i<numPoints; i++)
        {
            printf("dense Plate process: %d (%d) %.2f%%\r", i, numPoints, 100*float(i)/numPoints); 
            fflush(stdout);
//            cout << "Using sparse vote." << endl;
            plate_saliency = sparsePlate(0, i*2);

            if (vn.squaredNorm() > 0) {vn /= vn.norm(); }// normalize
            O = points.col(i).cast<float>(); // in unit of mm of float

            // cotangent
            vt << sparsePlate.col(i*2 + 1).tail(3).transpose();
            if (vt.squaredNorm() > 0) { vt /= vt.norm(); } //normalize

            U << sparseStick.col(i).tail(3), sparsePlate.col(i*2).tail(3);
            Np = U*U.transpose();

//            cout << "Theta sets for " << i << " : ";
            // all votee's are j's
            for (unsigned int j=0; j<numPoints; j++)
            {
                if(j == i) continue;
                P = points.col(j).cast<float>();
                v = P - O;
                l = v.norm();
                scaled_dist = l/sigma;
                // only consider points reasonablly near
                if (scaled_dist<3 && l>0)
                {
                    v /= l; // normalize
                    vvt = v.dot(vt); // size of points
                    if (vvt<0) {vt *= -1;}

                    vn = vt.cross(v).cross(vt);
                    if(vn.squaredNorm() >0) {vn /= vn.norm();}
                    vvn = v.dot(vn);
                    if (vvn<0) {vn *= -1;}

                    theta = asin(vvn); // separation angle O->P
#if 0
                    if(theta != 0 && theta!=M_PI/2 && theta!=-M_PI/2){
                        cout << i << "->" << j << endl;
                        cout << "theta: " << theta << endl;
                        cout << "vn: " << vn << endl;
                        cout << "vt: " << vt << endl;
                        cout << "v: " << v << endl;
                        cout << "vvn: " << vvn << endl;
                        cout << "vvt: " << vvt << endl;
                    }
#endif              
                    if (fabs(theta) <= M_PI/4 || fabs(theta) >= 3*M_PI/4)
                    {
                        vc = vn*cos(2*theta) - vt*sin(2*theta);
                        // method 1
//                        DF = radial_decay_smooth(scaled_dist)*angular_decay(theta);
//                        Sp = DF*plate_saliency*(vc.transpose()*vc);

                        // method 2
                        float c = radial_decay_smooth(scaled_dist);
                        float r = vc.norm();
                        DF=exp( -(4*r*r*theta*theta + c/(r*r)) / (sigma*sigma) );
                        Sp = DF*(vc.transpose()*vc);

                        //field(i) += Sp; // should be wrong
                        field(j) += Sp;
//                        cout << theta/M_PI  << ' '<< "DF" << DF << ' ' << "refcos: " << vvt << "cos: " << cos(theta) << endl;


                        // update the log: who_receied_whos
                        if (who_receied_whos.find(j) != who_receied_whos.end())
                        { // there already
                            who_receied_whos[j].push_back(i);
                        }
                        else
                        { // create structure
                            vector<unsigned int> senders;
                            senders.reserve(numPoints);
                            senders.push_back(i);
                            who_receied_whos[j]=senders;
                        }


                        h_log[i].x += 1;
                        h_log[j].y += 1;

                    }
                }
            } // all neighbouring points
//            cout << endl << "Tensor Element origined from " << i << " finished." << endl;

        } // all points to be processed


    for(int i =0; i<numPoints; i++)
    {
        logf << i << " " << h_log[i].x << " " << h_log[i].y << endl;
    }
    free(h_log);
    logf.close();
    printf ("\n dense whotowhom log finished... \n");


#if 0
        cout << "Who received whos message: " << endl;
        for(map<unsigned int, vector<unsigned int> >::iterator iter = who_receied_whos.begin();
                    iter !=who_receied_whos.end();
                    ++ iter)
        {
            cout << iter->first << '\t' << iter->second.size()<< '\t';
            for (unsigned int i = 0; i<iter->second.size(); i++)
            {
                cout << iter->second[i] << " ";
            }
            cout << endl;
        }
#endif 
//        cout << "FIELDMAT: " << field.rows() << " x " << field.cols() << endl;
        return field;
    } // end of function stickdensevote

Matrix<Matrix3f,Dynamic,1> TensorVoting::sparse_ball_vote()
/** this function executes sparse voting using the initial ball tensors
// at the locations of the input points based on a voting radius
// inputs: 		point cloud, sigma
// outputs: 	preliminary sparse tensors */
{

	const unsigned int num_pts = points.cols(); //number of input points

	Matrix3f I;
	I.setIdentity();
	
    sparseTensors.resize(num_pts, 1);
	for (unsigned int i=0; i<num_pts; ++i)
	{
		sparseTensors(i) = I;
	}
	
	Matrix<float,1,3> coord_voter;
	Matrix<float,1,3> coord_votee;

    // log 
    size_t sizeLog = num_pts*sizeof(int2);
    h_log = (int2 *)malloc(sizeLog);
    bzero( h_log, sizeLog);
    printf ( "Start logging\n" );
    ofstream logf("log_number_of_whowhom_cpu.txt");

	for (unsigned int voter=0; voter<num_pts; ++voter)
	{
        printf("Sparse voting process: %d (%d) %.2f%%\r", voter, num_pts, 100*float(voter)/num_pts); 
            fflush(stdout);
		coord_voter << points(0,voter),points(1,voter),points(2,voter);
		for (unsigned int votee=0; votee<num_pts; ++votee)
		{
			if (voter != votee)
			{
				coord_votee << points(0,votee),points(1,votee),points(2,votee);
				Matrix<float,1,3> v = coord_votee - coord_voter;
				float l = v.squaredNorm();
				float z = sqrt(l)/sigma;
				if (l>0 && z<3)
				{
					Matrix<float,3,3> vv = v.transpose()*v;
					float norm_vv = vv.norm();
					if (norm_vv > 0)
					{	
						sparseTensors(votee) = sparseTensors(votee) + radial_decay_smooth(z)*(I - vv/norm_vv);
						//sparseTensors(votee) = sparseTensors(votee) + radial_decay_traditional(z)*(I - vv/norm_vv);
					}
                    h_log[voter].x += 1;
                    h_log[votee].y += 1;
				}		
			}
		}
	}

    for(int i =0; i<num_pts; i++)
    {
        logf << i << " " << h_log[i].x << " " << h_log[i].y << endl;
    }
    free(h_log);
    logf.close();
    printf ("\n log finished... \n");
	return sparseTensors;
}

void TensorVoting::sparse_tensor_split ()
/** this function splits the sparse tensors into their 3 components (stick, plate, ball)
// inputs: 		preliminary sparse tensors
// outputs: 	stick, plate, (ball) components. 
// the stick and plate information is returned in a vector of columns containing the saliency and then the appropriate vector,
// (4 x N, where N is the number of points)  */
{
	
	const unsigned int num_pts = sparseTensors.rows();

    sparseStick.resize(4, num_pts);
    sparsePlate.resize(4, num_pts*2);
    sparseBall.resize(num_pts);


	Vector3f abs_evalues;
	Vector3f eig_indices(0,0,0);
	
	for (unsigned int i=0; i<num_pts; ++i)
	{
        printf("Sparse tensor split process: %d (%d) %.2f%%\r", i, num_pts, 100*float(i)/num_pts); 
        fflush(stdout);

        // calculate eigenvalues from simplified algorithm




		// eigenvalues and eigenvectors come sorted already, but not by absolute value...
		SelfAdjointEigenSolver<Matrix3f> eigensolver(sparseTensors(i));
		abs_evalues = eigensolver.eigenvalues().array().abs();		
		//find indeces corresponding to sorted eigenvalues: 
		abs_evalues.maxCoeff(&eig_indices(0));
		abs_evalues.minCoeff(&eig_indices(2));
		for (unsigned int x=0; x<3; ++x)
		{
			if (x != eig_indices(0) && x!= eig_indices(2)) { eig_indices(1) = x; }
		}	
		// store relevant stick, plate, ball information:
		sparseStick(0,i) = eigensolver.eigenvalues()[eig_indices(0)] - eigensolver.eigenvalues()[eig_indices(1)];
		sparseStick.col(i).tail(3) = eigensolver.eigenvectors().col(eig_indices(0));
		sparsePlate(0,i*2) = eigensolver.eigenvalues()[eig_indices(1)] - eigensolver.eigenvalues()[eig_indices(2)];
		//sparsePlate.col(i).tail(3) = eigensolver.eigenvectors().col(eig_indices(2)); // TODO: add full plate voting result
		sparsePlate.col(i*2 + 0).tail(3) = eigensolver.eigenvectors().col(eig_indices(1)); // save the second eigen value vector here
		sparsePlate.col(i*2 + 1).tail(3) = eigensolver.eigenvectors().col(eig_indices(2)); // save the smallest eigen value vector here
		sparseBall(i) = eigensolver.eigenvalues()[eig_indices(2)];
	}
}


} // end of namespace

A* while talking distance, weather and traffic into consideration (Without ML/DL)
	Start:  3:17:50 AM
	End: 4:45:27 AM
	Time difference: 1 hour, 27 minutes, and 37 seconds

With Neural Networks:

    exit_edge_sequential:
        Model: Sequential
        Epochs: 15
        Accuracy: 0.976491999696671
        Precision: 0.9622811916132145
        F1 Score: 0.9687884611885905
        Time Taken in simulation:
            Start: 3:59:16 AM
            End: 4:57:14 AM
            Time Difference: 0 hours, 57 minutes, and 58 seconds.

    exit_edge_lstm
        Model: LSTM
        Epochs: 15
        Accuracy: 0.9818192158944415
        Precision: 0.9738332929424465
        F1 Score: 0.9768116229772457
        Time Taken in simulation:
            Start: 10:35:34 PM
            End: 12:19:33 AM
            Time Difference: 1 hour, 43 minutes, and 58 seconds

    exit_edge_autoencoder:
        Model: AutoEncoder
        Epochs: 15
        Accuracy: 0.9773451126109047
        Precision: 0.9681482819911614
        F1 Score: 0.9718460016719759
        Time Taken in simulation:
            Start: 2:47:53 AM
            End: 4:49:38 AM
            Time Difference: 2 hours and 1 minute

    exit_edge_transformer
        Model: Transformer
        Epochs: 15
        Accuracy: 0.9806817320087965
        Precision: 0.973326229675175
        F1 Score: 0.9759148305844466
        Time Taken in simulation:
            Start: 3:00:56 AM
            End: 4:59:05 AM
            Time Difference: 1 hour, 58 minutes, and 9 seconds
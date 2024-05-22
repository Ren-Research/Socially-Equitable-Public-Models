def add_general_args(parser):
    parser.add_argument("--n_epochs", type=int, default=100,
                        help="Total training iterations")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Choose carefully based on gpu memory")
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--train_ratio", type=float, default=0.67)
    parser.add_argument("--training", action='store_true', help='Training')
    parser.add_argument("--q_idx", type=float, default=1, help="q+1")
    parser.add_argument("--baseline", action='store_true', help="Build baseline (plain PM)")
    parser.add_argument("--model_path", type=str, help="Path of the trained public model for evaluation")
    parser.add_argument("--diff_lambda", action='store_true', help="In data center application, use different lambda between groups")
    parser.add_argument("--diff_group_dist", action='store_true', help="Run with different groups distribution")
    args = parser.parse_args()
    return args